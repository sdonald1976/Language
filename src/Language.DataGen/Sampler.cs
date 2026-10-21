using Language.Core.Ir;
using Language.Core.Text;

namespace Language.DataGen;

/// <summary>Grammatical facts about a sampled mention, used for pronoun agreement.</summary>
sealed record MentionInfo(NounClass Class, char? Gender, bool Plural);

/// <summary>
/// Samples random IR v0 trees. Sentences are realized later by the rule-based realizer, so every
/// (text, IR) pair is correct by construction. With probability <see cref="JabberwockyRate"/> a sentence
/// replaces content words with invented words, which forces the model to rely on syntax and morphology.
/// </summary>
public sealed class Sampler
{
    readonly Random _rng;
    readonly List<string> _nonsense;
    readonly Dictionary<Node, MentionInfo> _info = new(ReferenceEqualityComparer.Instance);
    int _e, _m;
    bool _jabber;

    public double JabberwockyRate { get; init; } = 0.4;
    public double NonsenseWordRate { get; init; } = 0.5;

    public Sampler(Random rng, List<string> nonsensePool)
    {
        _rng = rng;
        _nonsense = nonsensePool;
    }

    double R() => _rng.NextDouble();
    T Pick<T>(IReadOnlyList<T> a) => a[_rng.Next(a.Count)];
    string NewE() => $"e{++_e}";
    string NewM() => $"m{++_m}";

    /// <summary>Invented words used in the last sampled tree (for lexicon-hint augmentation).</summary>
    public Dictionary<string, string> UsedNonsense { get; } = new();

    string Content(string real, string pos)
    {
        if (!_jabber || R() >= NonsenseWordRate) return real;
        var w = Pick(_nonsense);
        UsedNonsense[w] = pos;
        return w;
    }

    public Node Sample()
    {
        _info.Clear();
        UsedNonsense.Clear();
        _e = _m = 0;
        _jabber = R() < JabberwockyRate;

        var x = R();
        Node root;
        if (x < 0.45) root = new Node("assert").Child(TopClause(Fin.Finite));
        else if (x < 0.58) root = new Node("ynq").Child(Event(Fin.Finite, 0, allowRelations: R() < 0.5));
        else if (x < 0.71) root = new Node("whq").Child(WhQuestion());
        else if (x < 0.82) root = new Node("command").Child(Event(Fin.Imperative, 0, allowRelations: true));
        else if (x < 0.93) root = Define();
        else root = new Node("assert").Child(HedgedDefine());

        AddPronouns(root);
        FlagAmbiguity(root);
        return Schema.Renumber(Schema.Canonicalize(root));
    }

    // ---------------------------------------------------------------- ambiguity (IR-v0 §8)

    static readonly HashSet<string> Universal = ["every", "each", "all"];
    static readonly HashSet<string> Indefinite = ["a", "some", "two", "three", "many", "several"];

    /// <summary>Adds :ambig when the realized string has a second grammatical analysis:
    /// a PP right after a VP-internal common noun can attach to the noun or the verb ("saw the man with the telescope"),
    /// and a universal and an indefinite co-argument can scope either way ("Every child read a book").</summary>
    static void FlagAmbiguity(Node root)
    {
        var attachment = false;
        var scope = false;
        foreach (var e in root.Descendants().Where(d => d.IsEvent))
        {
            var obls = e.Props.Where(p => Schema.IsOblique(p.Role) && p.Node is not null).Select(p => p.Node!).ToList();
            var before = e.Get("xcomp") is { } x ? Rightmost(x) : e.Get("obj") ?? e.Get("iobj") ?? (e.Get("attr") is { IsMention: true } a ? a : null);
            if (obls.Count > 0 && IsAttachmentSite(before)) attachment = true;
            for (var i = 0; i + 1 < obls.Count; i++)
                if (IsAttachmentSite(Rightmost(obls[i]))) attachment = true;
            foreach (var p in e.Props.Where(p => p.Role is "obj" or "iobj" || Schema.IsOblique(p.Role)))
                if (p.Node is { } m && m.IsMention && m.Props.Any(q => Schema.IsOblique(q.Role)) && !e.Has("there")) attachment = true;

            var dets = e.Props.Where(p => p.Role is "subj" or "obj" or "iobj").Select(p => p.Node?.Atom("det")).ToList();
            if (dets.Any(d => d is not null && Universal.Contains(d)) && dets.Any(d => d is not null && Indefinite.Contains(d))) scope = true;
        }
        if (attachment) root.Add("ambig", "attachment");
        if (scope) root.Add("ambig", "scope");
    }

    static bool IsAttachmentSite(Node? m) =>
        m is { IsMention: true } && !m.Has("pron") && !m.Has("propn") && !m.Has("wh") && !m.Has("rel");

    static Node? Rightmost(Node n)
    {
        if (n.IsMention)
        {
            var last = n.Props.LastOrDefault(p => Schema.IsOblique(p.Role) && p.Node is not null);
            return last is not null ? Rightmost(last.Node!) : n;
        }
        var obl = n.Props.LastOrDefault(p => Schema.IsOblique(p.Role) && p.Node is not null);
        if (obl is not null) return Rightmost(obl.Node!);
        if (n.Get("xcomp") is { } x) return Rightmost(x);
        return n.Get("obj") ?? n.Get("iobj");
    }

    enum Fin { Finite, Imperative, NonFinite }

    Node TopClause(Fin fin)
    {
        if (R() < 0.12)
        {
            var conj = Pick(new[] { "and", "and", "but", "or" });
            return new Node(conj).Child(Event(fin, 0, false)).Child(Event(fin, 0, false));
        }
        return Event(fin, 0, allowRelations: true);
    }

    // ---------------------------------------------------------------- events

    Node Event(Fin fin, int depth, bool allowRelations, string? forceFrame = null)
    {
        var x = R();
        string frame = forceFrame ?? (x < 0.22 ? "I" : x < 0.6 ? "T" : x < 0.68 ? "D" : x < 0.75 && depth < 2 ? "C" : x < 0.82 && depth < 2 ? "X" : x < 0.95 ? "copula" : "there");
        if (fin != Fin.Finite && frame == "there") frame = "copula";

        Node e;
        if (frame is "copula" or "there") e = Copula(frame == "there", fin);
        else
        {
            var candidates = Vocab.Verbs.Where(v => v.Frames.Contains(frame[0])).ToList();
            var verb = Pick(candidates);
            e = new Node(Content(verb.Lemma, "v"), NewE());
            var passive = frame == "T" && fin == Fin.Finite && R() < 0.12;

            if (fin == Fin.Imperative) e.Add("subj", Implicit());
            else if (fin == Fin.Finite) e.Add("subj", Mention(verb.AnimateSubject && !passive ? Animacy.Animate : Animacy.Any, depth));

            if (passive)
            {
                e.Add("voice", "pass");
                if (R() < 0.6) e.Add("obl-by", Mention(Animacy.Animate, depth));
            }
            else switch (frame)
            {
                case "T": e.Add("obj", Mention(Animacy.Any, depth)); break;
                case "D":
                    if (R() < 0.6) { e.Add("iobj", Mention(Animacy.Animate, depth)); e.Add("obj", Mention(Animacy.Thing, depth)); }
                    else { e.Add("obj", Mention(Animacy.Thing, depth)); e.Add("obl-to", Mention(Animacy.Animate, depth)); }
                    break;
                case "C":
                    if (verb.Frames.Contains('D') && R() < 0.5) e.Add("iobj", Mention(Animacy.Animate, depth));
                    e.Add("comp", Event(Fin.Finite, depth + 1, false));
                    break;
                case "X":
                    e.Add("xcomp", Event(Fin.NonFinite, depth + 1, false, forceFrame: R() < 0.5 ? "I" : "T"));
                    break;
            }
        }

        if (fin != Fin.NonFinite || R() < 0.2) Features(e, fin);
        if (R() < 0.25) e.Add(Oblique(), Mention(Animacy.Place, depth));
        Adverbs(e, fin);

        if (allowRelations && depth == 0 && R() < 0.3)
        {
            var rel = Pick(new[] { "condition", "condition", "unless", "cause", "cause", "concession", "when", "before", "after", "while", "until", "purpose" });
            if (rel == "purpose") e.Add("purpose", Event(Fin.NonFinite, depth + 1, false, forceFrame: R() < 0.5 ? "I" : "T"));
            else e.Add(rel, Event(Fin.Finite, depth + 1, false));
        }
        return e;
    }

    Node Copula(bool there, Fin fin)
    {
        var e = new Node("be", NewE());
        if (there)
        {
            e.Flag("there");
            e.Add("subj", Mention(Animacy.Any, 1, indefinite: true));
            e.Add("obl-" + Pick(Vocab.PlacePreps), Mention(Animacy.Place, 1));
            return e;
        }
        if (fin == Fin.Imperative) e.Add("subj", Implicit());
        else if (fin == Fin.Finite) e.Add("subj", Mention(Animacy.Any, 1));
        if (R() < 0.15) e.Add("attr", new Node(Pick(Vocab.Adjectives)).Add("adv", Pick(Vocab.Intensifiers)));
        else if (R() < 0.75) e.Add("attr", Content(Pick(Vocab.Adjectives), "adj"));
        else e.Add("attr", Mention(Animacy.Any, 1, indefinite: true, bare: true));
        return e;
    }

    void Features(Node e, Fin fin)
    {
        if (fin == Fin.Finite)
        {
            if (R() < 0.12) e.Add("mod", Pick(Schema.Modals));
            else e.Add("tense", Pick(new[] { "past", "past", "past", "pres", "pres", "pres", "fut" }));
        }
        var a = fin == Fin.Imperative ? 1 : R();
        if (a < 0.12) e.Add("aspect", "prog");
        else if (a < 0.21) e.Add("aspect", "perf");
        else if (a < 0.24) e.Add("aspect", "perfprog");
        if (R() < 0.25)
        {
            e.Add("pol", "neg");
            if (fin == Fin.Finite && R() < 0.12) e.Add("negword", "never");
        }
    }

    void Adverbs(Node e, Fin fin)
    {
        var x = R();
        if (x < 0.1) e.Add("adv", Content(Pick(Vocab.MannerAdverbs), "adv"));
        else if (x < 0.18) e.Add("adv", Pick(Vocab.TimeAdverbs));
        else if (x < 0.23 && fin == Fin.Finite && !e.Has("negword")) e.Add("adv", Pick(Vocab.FrequencyAdverbs));
        else if (x < 0.26) e.Add("adv", new Node(Pick(Vocab.MannerAdverbs)).Add("adv", Pick(Vocab.Intensifiers)));
    }

    string Oblique() => "obl-" + (R() < 0.7 ? Pick(Vocab.PlacePreps) : Pick(Vocab.OtherPreps));

    Node WhQuestion()
    {
        var e = Event(Fin.Finite, 0, allowRelations: R() < 0.2);
        var options = e.Props.Where(p => p.Node is { IsMention: true } && !p.Node.Has("implicit") && p.Role != "iobj"
                                         && (Schema.IsArgumentRole(p.Role) || p.Role == "attr") && !p.Node.Has("rel")).ToList();
        if (options.Count == 0 || R() < 0.25)
        {
            e.Add("wh-adv", Pick(Schema.WhAdverbs));
            return e;
        }
        var target = Pick(options);
        var info = _info.GetValueOrDefault(target.Node!);
        var human = info?.Class is NounClass.Human;
        Node wh;
        if (R() < 0.2 && !target.Node!.Has("pron") && !target.Node.Has("propn"))
            wh = new Node(target.Node.Head, NewM()).Add("det", "which").Flag("wh");
        else
            wh = new Node(human ? "who" : "what", NewM()).Flag("wh");
        var i = e.Props.IndexOf(target);
        e.Props[i] = target with { Node = wh };
        return e;
    }

    // ---------------------------------------------------------------- definitions

    Node Define()
    {
        var verbTerm = R() < 0.25;
        var term = verbTerm ? new Node(Pick(_nonsense), NewE()) : new Node(Pick(_nonsense), NewM());
        UsedNonsense[term.Head] = verbTerm ? "v" : "n";
        Node body;
        string marker;
        if (verbTerm)
        {
            body = Event(Fin.NonFinite, 1, false, forceFrame: R() < 0.5 ? "I" : "T");
            marker = R() < 0.8 ? "means" : "refers-to";
        }
        else
        {
            var article = R() < 0.4;
            if (article) term.Add("det", "a");
            body = Mention(Animacy.Any, 1, indefinite: article, bare: !article);
            if (article && body.Atom("num") == "pl")
            {
                body.Remove("num");
                body.Set("det", "a");
            }
            if (R() < 0.3) body = new Node("kind", NewM()).Add("det", "a").Add("obl-of", Bare(body));
            marker = article ? "is" : Pick(new[] { "means", "means", "refers-to" });
        }
        return new Node("define").Add("term", term).Add("def", body).Add("marker", marker);
    }

    static Node Bare(Node m)
    {
        m.Remove("det");
        return m;
    }

    Node HedgedDefine()
    {
        var verb = Pick(new[] { "think", "believe", "say", "hope" });
        var e = new Node(verb, NewE()).Add("tense", Pick(new[] { "pres", "past" }));
        e.Add("subj", R() < 0.6 ? Pronoun("I") : Mention(Animacy.Animate, 1));
        e.Add("comp", Define());
        return e;
    }

    // ---------------------------------------------------------------- mentions

    enum Animacy { Any, Animate, Thing, Place }

    Node Implicit() => new Node("you", NewM()).Flag("pron").Flag("implicit");

    Node Pronoun(string lemma)
    {
        var n = new Node(lemma, NewM()).Flag("pron");
        var p = Words.Pronouns[lemma];
        _info[n] = new MentionInfo(p.Gender == 'n' ? NounClass.Thing : NounClass.Human, p.Gender is 'm' or 'f' ? p.Gender : null, p.Plural);
        return n;
    }

    Node Mention(Animacy anim, int depth, bool indefinite = false, bool bare = false)
    {
        var x = R();
        if (!bare && !indefinite && x < 0.12) return Pronoun(Pick(new[] { "I", "you", "we", "I", "you" }));
        if (!bare && !indefinite && x < 0.3 && anim is Animacy.Any or Animacy.Animate)
        {
            var name = Pick(Vocab.Names);
            var n = new Node(name.Name, NewM()).Flag("propn");
            _info[n] = new MentionInfo(NounClass.Human, name.Gender, false);
            return n;
        }

        var pool = anim switch
        {
            Animacy.Animate => Vocab.Nouns.Where(n => n.Class is NounClass.Human or NounClass.Animal).ToArray(),
            Animacy.Thing => Vocab.Nouns.Where(n => n.Class is NounClass.Thing).ToArray(),
            Animacy.Place => Vocab.Nouns.Where(n => n.Class is NounClass.Place or NounClass.Thing).ToArray(),
            _ => Vocab.Nouns,
        };
        var entry = Pick(pool);
        var head = Content(entry.Lemma, "n");
        var cls = head == entry.Lemma ? entry.Class : NounClass.Unknown;
        var m = new Node(head, NewM());
        var mass = cls == NounClass.Mass;
        var plural = !mass && R() < 0.25;

        if (indefinite)
        {
            if (!plural && !mass) m.Add("det", "a");
            else if (R() < 0.5) m.Add("det", "some");
        }
        else if (!bare && R() < 0.1) m.Add("poss", Possessor());
        else if (!bare)
        {
            var det = Pick(Vocab.Determiners);
            if (plural && det is "a" or "every" or "each") det = Pick(new[] { "the", "some", "these", "those", "two", "many" });
            if (!plural && det is "these" or "those" or "two" or "many") det = "the";
            if (mass && det is "a" or "every" or "each") det = "the";
            if (det == "this" && plural) det = "these";
            if (det == "that" && plural) det = "those";
            if (!(plural && R() < 0.2) && !(mass && R() < 0.4)) m.Add("det", det);
        }
        if (plural) m.Add("num", "pl");

        var a = R();
        if (a < 0.22) m.Add("adj", Content(Pick(Vocab.Adjectives), "adj"));
        else if (a < 0.26) { m.Add("adj", Pick(Vocab.Adjectives)); m.Add("adj", Pick(Vocab.Adjectives)); }
        else if (a < 0.29) m.Add("adj", new Node(Pick(Vocab.Adjectives)).Add("adv", Pick(Vocab.Intensifiers)));

        if (depth < 2 && R() < 0.05) m.Add("obl-" + Pick(Vocab.PlacePreps), Mention(Animacy.Place, depth + 1, bare: false));
        _info[m] = new MentionInfo(cls, entry.Gender, plural);

        if (depth < 1 && !bare && R() < 0.07 && m.Atom("det") is "the" or "a" or null) AddRelative(m, depth);
        return m;
    }

    Node Possessor()
    {
        if (R() < 0.5) return Pronoun(Pick(new[] { "I", "you", "we" }));
        var name = Pick(Vocab.Names);
        var n = new Node(name.Name, NewM()).Flag("propn");
        _info[n] = new MentionInfo(NounClass.Human, name.Gender, false);
        return n;
    }

    void AddRelative(Node host, int depth)
    {
        var verb = Pick(Vocab.Verbs.Where(v => v.Frames.Contains('T')).ToList());
        var e = new Node(Content(verb.Lemma, "v"), NewE());
        if (R() < 0.55)
        {
            e.Add("subj", host.Id!);
            e.Add("obj", Mention(Animacy.Any, depth + 2, bare: false));
        }
        else
        {
            e.Add("subj", Mention(Animacy.Animate, depth + 2));
            e.Add("obj", host.Id!);
        }
        e.Add("tense", Pick(new[] { "past", "pres" }));
        if (R() < 0.15) e.Add("pol", "neg");
        host.Add("rel", e);
    }

    // ---------------------------------------------------------------- pronouns and coreference

    /// <summary>Replaces some later mentions with third-person pronouns and labels :ref by the
    /// agreement and binding rules in docs/IR-v0.md §6.</summary>
    void AddPronouns(Node root)
    {
        var tries = R() < 0.45 ? 1 + (R() < 0.3 ? 1 : 0) : 0;
        for (var t = 0; t < tries; t++)
        {
            var slots = ArgumentSlots(root).Where(s => !s.Node.Has("pron") && !s.Node.Has("wh") && !s.Node.Has("implicit") && !s.Node.Has("rel")
                                                       && s.Node.Children.All(c => !c.IsMention || c.Has("pron"))).ToList();
            if (slots.Count < 2) return;
            var antecedent = Pick(slots).Node;
            if (!_info.TryGetValue(antecedent, out var ai)) continue;
            var lemma = PronounFor(ai);
            if (lemma is null) continue;
            var targets = slots.Where(s => !ReferenceEquals(s.Node, antecedent) && !Contains(s.Node, antecedent) && !Contains(antecedent, s.Node)).ToList();
            if (targets.Count == 0) continue;
            var target = Pick(targets);
            var original = target.Node;
            if (root.Descendants().Any(d => d.Atoms("ref").Contains(original.Id!))) continue;

            Node pron;
            bool reflexive = target.Event is not null && target.Role != "subj"
                             && ReferenceEquals(target.Event.Get("subj"), antecedent);
            if (target.Role != "subj" && R() < 0.3 && target.Node.IsMention && !target.Node.Has("propn"))
            {
                // Possessive pronoun on the target noun ("his car") instead of replacing it.
                if (target.Node.Has("poss")) continue;
                pron = Pronoun(lemma);
                target.Node.Remove("det");
                target.Node.Add("poss", pron);
                Label(root, pron, possessive: true);
                continue;
            }
            pron = Pronoun(lemma);
            if (reflexive) pron.Flag("refl");
            target.Replace(pron);
            Label(root, pron, possessive: false);
            if (!pron.Has("ref"))
                target.Replace(original); // a configuration we do not generate (see Label); keep the original
        }
    }

    string? PronounFor(MentionInfo i)
    {
        if (i.Plural) return "they";
        return i.Class switch
        {
            NounClass.Human => i.Gender switch { 'm' => "he", 'f' => "she", _ => R() < 0.5 ? "he" : "she" },
            NounClass.Unknown => R() < 0.6 ? "it" : R() < 0.5 ? "he" : "she",
            _ => "it",
        };
    }

    static bool Compatible(string pron, MentionInfo i) => pron switch
    {
        "they" => i.Plural,
        "it" => !i.Plural && i.Class is not NounClass.Human,
        "he" => !i.Plural && (i.Class is NounClass.Human or NounClass.Unknown) && i.Gender is null or 'm',
        "she" => !i.Plural && (i.Class is NounClass.Human or NounClass.Unknown) && i.Gender is null or 'f',
        _ => false,
    };

    void Label(Node root, Node pron, bool possessive)
    {
        var all = ArgumentSlots(root).ToList();
        var mine = all.FirstOrDefault(s => ReferenceEquals(s.Node, pron));
        var candidates = new List<Node>();
        foreach (var d in root.Descendants())
        {
            if (!d.IsMention || ReferenceEquals(d, pron) || d.Has("pron") || d.Has("wh") || d.Has("implicit")) continue;
            if (Contains(d, pron) || Contains(pron, d)) continue;
            if (!_info.TryGetValue(d, out var info) || !Compatible(pron.Head, info)) continue;
            if (!possessive && mine.Event is not null)
            {
                var coArg = mine.Event.Props.Any(p => ReferenceEquals(p.Node, d) && Schema.IsArgumentRole(p.Role));
                if (pron.Has("refl") ? !coArg : coArg) continue; // binding Principles A and B
                // Principle C: a pronoun subject cannot corefer with a name inside its own clause.
                if (mine.Role == "subj" && Contains(mine.Event, d)) return;
            }
            candidates.Add(d);
        }
        if (pron.Has("refl"))
        {
            var subj = mine.Event?.Get("subj");
            if (subj is null || !candidates.Contains(subj)) { pron.Remove("refl"); return; }
            candidates = [subj];
        }
        if (candidates.Count == 0) pron.Add("ref", "none");
        else foreach (var c in candidates) pron.Add("ref", c.Id!);
    }

    static bool Contains(Node outer, Node inner) => !ReferenceEquals(outer, inner) && outer.Descendants().Any(d => ReferenceEquals(d, inner));

    sealed class Slot(Node owner, int index, string role, Node? ev)
    {
        public Node Owner => owner;
        public string Role => role;
        public Node? Event => ev;
        public Node Node => owner.Props[index].Node!;
        public void Replace(Node n) => owner.Props[index] = owner.Props[index] with { Node = n };
    }

    static IEnumerable<Slot> ArgumentSlots(Node root)
    {
        foreach (var d in root.Descendants())
            for (var i = 0; i < d.Props.Count; i++)
            {
                var p = d.Props[i];
                if (p.Node is { IsMention: true } && (Schema.IsArgumentRole(p.Role) || p.Role == "poss"))
                    yield return new Slot(d, i, p.Role, d.IsEvent ? d : null);
            }
    }
}
