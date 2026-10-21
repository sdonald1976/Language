using Language.Core.Ir;
using Language.Core.Text;

namespace Language.DataGen;

public sealed record UdToken(int Id, string Form, string Lemma, string Upos, Dictionary<string, string> Feats, int Head, string Deprel)
{
    public string Rel => Deprel.Split(':')[0];
    public string? Feat(string k) => Feats.GetValueOrDefault(k);
}

public sealed record UdSentence(string Id, string Text, List<UdToken> Tokens);

/// <summary>
/// Converts Universal Dependencies trees into IR v0.
/// Strict mode (training data, S2) rejects any sentence with a construction it cannot map faithfully,
/// and any sentence with third-person pronouns, whose reference UD does not annotate.
/// Lenient mode (baseline B0 reader, on automatic parses) skips what it cannot map instead of rejecting.
/// </summary>
public sealed class UdConverter(bool strict)
{
    public static IEnumerable<UdSentence> ReadConllu(string path)
    {
        string? id = null, text = null;
        var toks = new List<UdToken>();
        foreach (var raw in File.ReadLines(path).Append(""))
        {
            var line = raw.TrimEnd('\r');
            if (line.Length == 0)
            {
                if (toks.Count > 0 && text is not null) yield return new UdSentence(id ?? "", text, toks);
                id = text = null;
                toks = new List<UdToken>();
                continue;
            }
            if (line.StartsWith("# sent_id = ")) { id = line[12..]; continue; }
            if (line.StartsWith("# text = ")) { text = line[9..]; continue; }
            if (line.StartsWith('#')) continue;
            var c = line.Split('\t');
            if (c.Length < 8 || c[0].Contains('-') || c[0].Contains('.')) continue;
            var feats = c[5] == "_" ? new Dictionary<string, string>() :
                c[5].Split('|').Select(f => f.Split('=')).Where(f => f.Length == 2).ToDictionary(f => f[0], f => f[1]);
            toks.Add(new UdToken(int.Parse(c[0]), c[1], c[2], c[3], feats, int.Parse(c[6]), c[7]));
        }
    }

    sealed class Reject(string why) : Exception(why);

    List<UdToken> _t = new();
    int _e, _m;
    readonly Dictionary<int, Node> _mentionOf = new();

    public string? LastRejection { get; private set; }

    public Node? Convert(UdSentence s)
    {
        _t = s.Tokens;
        _e = _m = 0;
        _mentionOf.Clear();
        LastRejection = null;
        try
        {
            if (strict && _t.Count > 30) throw new Reject("too long");
            if (strict && _t.Any(t => t.Upos == "PRON" && t.Feat("PronType") == "Prs" && t.Feat("Person") == "3"))
                throw new Reject("3rd-person pronoun (reference not annotated in UD)");
            var root = _t.FirstOrDefault(t => t.Head == 0) ?? throw new Reject("no root");
            var question = s.Text.TrimEnd().EndsWith('?');
            var imperative = root.Feat("Mood") == "Imp" || (root.Upos == "VERB" && root.Feat("VerbForm") == "Inf" && !Kids(root).Any(k => k.Rel is "nsubj" or "aux" or "mark"));
            if (strict && !imperative && !Kids(root).Any(k => k.Rel is "nsubj" or "expl")) throw new Reject("finite clause without subject");
            var clause = ClauseWithCoordination(root, imperative ? Fin.Imperative : Fin.Finite);
            var hasWh = _t.Any(t => t.Feat("PronType") == "Int");
            if (strict && hasWh && !question) throw new Reject("wh-word without question mark");
            var act = imperative && !question ? "command" : question ? (hasWh ? "whq" : "ynq") : "assert";
            var ir = new Node(act).Child(clause);
            ir = Schema.Renumber(Schema.Canonicalize(ir));
            var errors = Validator.Validate(ir);
            if (errors.Count > 0) throw new Reject("invalid: " + errors[0]);
            var tokens = Tokenizer.Tokenize(s.Text);
            var anch = Anchoring.Compute(ir, tokens);
            if (anch.Unanchored.Count > 0 && strict) throw new Reject("unanchored: " + anch.Unanchored[0]);
            if (strict && Coverage.Uncovered(ir, tokens).FirstOrDefault() is { } missing) throw new Reject("content word not in IR: " + missing);
            return ir;
        }
        catch (Reject r)
        {
            LastRejection = r.Message;
            return null;
        }
    }

    enum Fin { Finite, Imperative, NonFinite }

    IEnumerable<UdToken> Kids(UdToken h) => _t.Where(t => t.Head == h.Id);

    void Unsupported(string what)
    {
        if (strict) throw new Reject(what);
    }

    Node ClauseWithCoordination(UdToken head, Fin fin)
    {
        var conjs = Kids(head).Where(k => k.Rel == "conj" && IsPredicate(k)).ToList();
        var first = Event(head, fin);
        if (conjs.Count == 0) return first;
        var cc = Kids(conjs[0]).FirstOrDefault(k => k.Rel == "cc")?.Lemma.ToLowerInvariant() ?? "and";
        if (!Schema.Coordinators.Contains(cc)) { Unsupported("coordinator " + cc); cc = "and"; }
        var coord = new Node(cc).Child(first);
        foreach (var c in conjs)
        {
            if (strict && !Kids(c).Any(k => k.Rel is "nsubj")) throw new Reject("conjoined clause without own subject");
            coord.Child(Event(c, fin));
        }
        return coord;
    }

    static bool IsPredicate(UdToken t) => t.Upos is "VERB" or "AUX" || t.Upos is "ADJ" or "NOUN";

    static readonly Dictionary<string, string> Marks = new()
    {
        ["if"] = "condition", ["unless"] = "unless", ["because"] = "cause", ["although"] = "concession", ["though"] = "concession",
        ["when"] = "when", ["before"] = "before", ["after"] = "after", ["while"] = "while", ["until"] = "until", ["till"] = "until",
    };
    static readonly HashSet<string> ModalLemmas = ["can", "could", "may", "might", "must", "should", "would", "shall"];

    Node Event(UdToken v, Fin fin)
    {
        var kids = Kids(v).ToList();
        var cop = kids.FirstOrDefault(k => k.Rel == "cop");
        var e = new Node(cop is not null ? "be" : v.Lemma.ToLowerInvariant(), $"e{++_e}");
        if (cop is null && v.Upos is not ("VERB" or "AUX")) throw new Reject($"non-verbal predicate without copula: {v.Form}");

        // Verb group.
        var auxes = kids.Where(k => k.Rel == "aux").OrderBy(k => k.Id).ToList();
        var passive = kids.Any(k => k.Deprel == "aux:pass");
        string? mod = null;
        var fut = false;
        foreach (var a in auxes)
        {
            var l = a.Lemma.ToLowerInvariant();
            if (l is "will" or "'ll" or "wo") fut = true;
            else if (ModalLemmas.Contains(l)) mod = l;
            else if (l is "have" or "be" or "do" or "to") { }
            else Unsupported("aux " + l);
        }
        var finite = (IEnumerable<UdToken>)auxes.Where(a => a.Feat("VerbForm") == "Fin");
        if (cop is not null && cop.Feat("VerbForm") == "Fin") finite = finite.Append(cop);
        if (v.Feat("VerbForm") == "Fin") finite = finite.Append(v);
        var finiteTok = finite.OrderBy(t => t.Id).FirstOrDefault();
        var perf = auxes.Any(a => a.Lemma == "have") && (v.Feat("Tense") == "Past" && v.Feat("VerbForm") == "Part" || passive || cop is not null && cop.Form.ToLowerInvariant() == "been" || auxes.Any(a => a.Form.ToLowerInvariant() == "been"));
        var prog = (v.Feat("VerbForm") == "Part" && v.Feat("Tense") == "Pres" && auxes.Any(a => a.Lemma == "be"))
                   || (passive && auxes.Any(a => a.Form.ToLowerInvariant() == "being"))
                   || (cop is not null && cop.Form.ToLowerInvariant() == "being");
        if (auxes.Any(a => a.Lemma == "be" && a.Form.ToLowerInvariant() == "going")) Unsupported("going to");

        if (fin == Fin.Finite)
        {
            if (mod is not null) e.Add("mod", mod);
            else if (fut) e.Add("tense", "fut");
            else if (finiteTok?.Feat("Tense") is { } t) e.Add("tense", t == "Past" ? "past" : "pres");
            else throw new Reject("no finite verb");
        }
        if (perf && prog) e.Add("aspect", "perfprog");
        else if (perf) e.Add("aspect", "perf");
        else if (prog) e.Add("aspect", "prog");
        var neg = kids.FirstOrDefault(k => k.Rel == "advmod" && k.Lemma.ToLowerInvariant() is "not" or "n't" or "never");
        if (neg is not null)
        {
            e.Add("pol", "neg");
            if (neg.Lemma.ToLowerInvariant() == "never") e.Add("negword", "never");
        }
        if (passive) e.Add("voice", "pass");
        if (kids.Any(k => k.Rel == "expl" && k.Lemma.ToLowerInvariant() == "there")) e.Flag("there");

        // Copular complement.
        if (cop is not null)
        {
            if (v.Upos is "ADJ") AddAdjective(e, "attr", v, predicate: true);
            else if (v.Upos is "NOUN" or "PROPN" or "PRON" or "NUM") e.Add("attr", Mention(v, allowRel: true, predicate: true));
            else Unsupported("copular complement " + v.Upos);
        }

        foreach (var k in kids.OrderBy(k => k.Id))
        {
            switch (k.Deprel)
            {
                case "nsubj" or "nsubj:pass":
                    if (k.Upos is "VERB") { Unsupported("clausal subject"); break; }
                    e.Add("subj", Mention(k, allowRel: true));
                    break;
                case "obj": e.Add("obj", Mention(k, allowRel: true)); break;
                case "iobj": e.Add("iobj", Mention(k, allowRel: true)); break;
                case "obl" or "obl:agent":
                {
                    var prep = Kids(k).FirstOrDefault(x => x.Rel == "case");
                    if (prep is null) { Unsupported("obl without preposition"); break; }
                    var p = prep.Lemma.ToLowerInvariant();
                    if (!p.All(char.IsLetter)) { Unsupported("odd preposition"); break; }
                    e.Add("obl-" + p, Mention(k, allowRel: true));
                    break;
                }
                case "advmod":
                    if (ReferenceEquals(k, neg)) break;
                    if (cop is not null && v.Upos == "ADJ" && Intensifiers.Contains(k.Lemma.ToLowerInvariant())) break; // went to :attr
                    if (k.Feat("PronType") == "Int") { e.Add("wh-adv", k.Lemma.ToLowerInvariant()); break; }
                    if (k.Upos != "ADV" || Kids(k).Any(x => x.Rel != "advmod")) { Unsupported("complex advmod"); break; }
                    var inner = Kids(k).Where(x => x.Rel == "advmod").ToList();
                    if (inner.Count == 0) e.Add("adv", k.Lemma.ToLowerInvariant());
                    else e.Add("adv", Modifier(k.Lemma.ToLowerInvariant(), inner));
                    break;
                case "xcomp":
                    if (k.Upos == "VERB" && Kids(k).Any(x => x.Rel == "mark" && x.Lemma == "to") && !Kids(k).Any(x => x.Rel == "nsubj"))
                        e.Add("xcomp", Event(k, Fin.NonFinite));
                    else Unsupported("xcomp " + k.Upos);
                    break;
                case "ccomp":
                    if (Kids(k).Any(x => x.Rel == "nsubj" || x.Rel == "expl") && !Kids(k).Any(x => x.Rel == "mark" && x.Lemma is not "that"))
                        e.Add("comp", ClauseWithCoordination(k, Fin.Finite));
                    else Unsupported("ccomp");
                    break;
                case "advcl":
                {
                    var mark = Kids(k).Where(x => x.Rel == "mark").Select(x => x.Lemma.ToLowerInvariant()).ToList();
                    if (mark.Count == 1 && mark[0] == "to" && !Kids(k).Any(x => x.Rel == "nsubj")) e.Add("purpose", Event(k, Fin.NonFinite));
                    else if (mark.Count == 1 && Marks.TryGetValue(mark[0], out var rel) && Kids(k).Any(x => x.Rel is "nsubj" or "expl" or "nsubj:pass")) e.Add(rel, Event(k, Fin.Finite));
                    else Unsupported("advcl " + string.Join("+", mark));
                    break;
                }
                case "aux" or "aux:pass" or "cop" or "punct" or "expl" or "mark" or "cc": break;
                case "conj" when IsPredicate(k): break;
                case "discourse" or "vocative" or "parataxis" or "dep" or "list" or "reparandum" or "orphan" or "goeswith":
                    Unsupported(k.Deprel);
                    break;
                default:
                    if (cop is not null && (k.Rel is "det" or "amod" or "nmod" or "case" or "compound" or "nummod" or "acl" or "advmod")) break; // belong to the complement
                    Unsupported(k.Deprel);
                    break;
            }
        }
        if (fin == Fin.Imperative && !e.Has("subj")) e.Add("subj", new Node("you", $"m{++_m}").Flag("pron").Flag("implicit"));
        if (fin != Fin.Finite) e.Remove("tense");
        return e;
    }

    Node Modifier(string head, List<UdToken> advs)
    {
        var n = new Node(head);
        foreach (var a in advs) n.Add("adv", a.Lemma.ToLowerInvariant());
        return n;
    }

    static readonly HashSet<string> Intensifiers = ["very", "too", "so", "quite", "really", "extremely", "rather", "pretty", "fairly", "incredibly", "super", "completely", "totally", "somewhat"];

    /// <summary>Adds an adjective as an atom, or as (adj :adv intensifier) when it has degree modifiers.</summary>
    void AddAdjective(Node target, string role, UdToken a, bool predicate = false)
    {
        if (a.Feat("Degree") is "Cmp" or "Sup") Unsupported("comparative");
        var kids = Kids(a).ToList();
        var advs = kids.Where(k => k.Rel == "advmod" && k.Upos == "ADV" && Intensifiers.Contains(k.Lemma.ToLowerInvariant())).ToList();
        if (!predicate && kids.Any(k => !advs.Contains(k) && k.Rel != "punct")) Unsupported("adjective with dependents");
        var head = a.Form.ToLowerInvariant();
        if (advs.Count == 0) target.Add(role, head);
        else target.Add(role, Modifier(head, advs));
    }

    /// <summary>Dependents of a predicate nominal that belong to the clause, handled by <see cref="Event"/>.</summary>
    static readonly HashSet<string> ClauseRels = ["nsubj", "cop", "aux", "mark", "advcl", "obl", "ccomp", "xcomp", "expl", "advmod", "parataxis", "discourse"];

    Node Mention(UdToken n, bool allowRel, bool predicate = false)
    {
        if (n.Upos is "VERB" or "ADJ" or "ADV" or "AUX") throw new Reject($"non-nominal argument {n.Upos} '{n.Form}'");
        var kids = Kids(n).OrderBy(k => k.Id)
            .Where(k => !(predicate && (ClauseRels.Contains(k.Rel) || k.Rel == "conj" && IsPredicate(k) || k.Rel == "cc"))).ToList();
        string head;
        var pronLemma = n.Upos == "PRON" ? Words.PronounLemma(n.Form) : null;
        if (pronLemma is not null && n.Feat("PronType") == "Prs") head = pronLemma;
        else if (n.Upos == "PROPN")
        {
            var flat = kids.Where(k => k.Rel == "flat").Select(k => k.Form);
            head = string.Join("_", new[] { n.Form }.Concat(flat));
        }
        else if (n.Upos == "PRON" && n.Feat("PronType") is "Int" or "Rel") head = n.Form.ToLowerInvariant();
        else head = n.Upos == "NUM" ? n.Form.ToLowerInvariant() : n.Lemma.ToLowerInvariant();
        if (head.Any(c => char.IsWhiteSpace(c) || c is '(' or ')' or '"')) throw new Reject("odd head");

        var m = new Node(head, $"m{++_m}");
        _mentionOf[n.Id] = m;
        if (pronLemma is not null && n.Feat("PronType") == "Prs")
        {
            m.Flag("pron");
            if (n.Feat("Reflex") == "Yes") m.Flag("refl");
        }
        if (n.Upos == "PROPN") m.Flag("propn");
        if (n.Feat("PronType") == "Int") m.Flag("wh");
        if (n.Feat("Number") == "Plur" && n.Upos is "NOUN" or "PROPN") m.Add("num", "pl");

        foreach (var k in kids)
        {
            switch (k.Deprel)
            {
                case "det":
                    var d = k.Lemma.ToLowerInvariant();
                    m.Add("det", d == "an" ? "a" : d);
                    if (k.Feat("PronType") == "Int") m.Flag("wh");
                    break;
                case "det:poss" or "nmod:poss":
                    if (Kids(k).Any(x => x.Rel == "case") || k.Upos == "PRON" || k.Upos is "PROPN" or "NOUN") m.Add("poss", Mention(k, false));
                    else Unsupported("possessive");
                    break;
                case "nummod": m.Add("det", k.Form.ToLowerInvariant()); break;
                case "amod":
                    if (k.Upos == "ADJ") AddAdjective(m, "adj", k);
                    else Unsupported("amod " + k.Upos);
                    break;
                case "compound":
                    if (k.Upos is "NOUN" && !Kids(k).Any()) m.Add("compound", k.Lemma.ToLowerInvariant());
                    else Unsupported("compound " + k.Upos);
                    break;
                case "nmod":
                {
                    var prep = Kids(k).FirstOrDefault(x => x.Rel == "case");
                    if (prep is null || !prep.Lemma.All(char.IsLetter)) { Unsupported("nmod without preposition"); break; }
                    m.Add("obl-" + prep.Lemma.ToLowerInvariant(), Mention(k, allowRel));
                    break;
                }
                case "acl:relcl":
                    if (!allowRel) { Unsupported("nested relative"); break; }
                    m.Add("rel", Relative(k, m));
                    break;
                case "case" or "punct" or "flat" or "cc": break;
                case "conj":
                    Unsupported("noun coordination");
                    break;
                default:
                    Unsupported("mention dep " + k.Deprel);
                    break;
            }
        }
        if (kids.Any(k => k.Rel == "conj") && !strict)
        {
            // Lenient mode: build a coordination of the head and its conjuncts.
            var coord = new Node(kids.FirstOrDefault(k => k.Rel == "conj") is { } c && Kids(c).FirstOrDefault(x => x.Rel == "cc") is { } cc ? cc.Lemma.ToLowerInvariant() : "and");
            coord.Child(m);
            foreach (var c2 in kids.Where(k => k.Rel == "conj")) coord.Child(Mention(c2, allowRel));
            return coord;
        }
        return m;
    }

    /// <summary>Relative clause: the relative pronoun's position becomes the gap (host id); "that/who/which" itself is dropped.</summary>
    Node Relative(UdToken verb, Node host)
    {
        var relPron = _t.FirstOrDefault(t => t.Feat("PronType") == "Rel" && IsWithin(t, verb));
        var e = Event(verb, Fin.Finite);
        if (relPron is null) throw new Reject("relative clause without relative pronoun");
        if (!_mentionOf.TryGetValue(relPron.Id, out var gapNode)) throw new Reject("relative pronoun position unsupported");
        foreach (var d in e.Descendants())
            for (var i = 0; i < d.Props.Count; i++)
                if (ReferenceEquals(d.Props[i].Node, gapNode))
                    d.Props[i] = d.Props[i] with { Node = null, Atom = host.Id };
        return e;
    }

    bool IsWithin(UdToken t, UdToken ancestor)
    {
        var cur = t;
        for (var guard = 0; guard < 50 && cur.Head != 0; guard++)
        {
            if (cur.Head == ancestor.Id) return true;
            cur = _t.First(x => x.Id == cur.Head);
        }
        return false;
    }
}
