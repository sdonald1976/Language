using Language.Core.Ir;
using Language.Core.Text;

namespace Language.Core.Realize;

/// <summary>Surface choices that do not change meaning. The realizer may vary these; the IR never encodes them.</summary>
public sealed record RealizerOptions
{
    public bool Contractions { get; init; }
    /// <summary>Put subordinate clauses (if/because/although/when...) before the main clause.</summary>
    public bool FrontSubordinate { get; init; }
    /// <summary>Express :cause as "X, so Y" instead of "Y because X".</summary>
    public bool CauseAsSo { get; init; }
    public bool CauseAsSince { get; init; }
    public bool ConcessionAsThough { get; init; }
    public bool DropComplementizer { get; init; }
    public bool RelativeWho { get; init; }

    public static readonly RealizerOptions Default = new();

    public static RealizerOptions Sample(Random rng) => new()
    {
        Contractions = rng.NextDouble() < 0.4,
        FrontSubordinate = rng.NextDouble() < 0.4,
        CauseAsSo = rng.NextDouble() < 0.15,
        CauseAsSince = rng.NextDouble() < 0.2,
        ConcessionAsThough = rng.NextDouble() < 0.3,
        DropComplementizer = rng.NextDouble() < 0.4,
        RelativeWho = rng.NextDouble() < 0.5,
    };
}

/// <summary>
/// Deterministic rule-based IR → English realizer. It is the always-faithful writing fallback (baseline B0),
/// and the synthetic data generator uses it to produce training sentences.
/// It only ever outputs words that come from the IR, plus closed-class function words.
/// </summary>
public sealed class Realizer
{
    readonly RealizerOptions _o;
    readonly Dictionary<string, Node> _ids = new();

    public Realizer(RealizerOptions? options = null) => _o = options ?? RealizerOptions.Default;

    public static string Realize(Node root, RealizerOptions? options = null) => new Realizer(options).Sentence(root);

    enum Mood { Decl, Ynq, Whq, Imp, Infinitive }
    enum Case { Nom, Acc, Poss }

    sealed record Ctx(Mood Mood, string? GapId = null);

    /// <summary>Realizes a single phrase: a noun phrase for a mention, a "to"-infinitive for an event.</summary>
    public static string Phrase(Node n, RealizerOptions? options = null)
    {
        var r = new Realizer(options);
        foreach (var d in n.Descendants())
            if (d.Id is not null) r._ids[d.Id] = d;
        return Tidy(n.IsEvent ? r.Infinitive(n, null) : r.NounPhrase(n, Case.Nom, null));
    }

    public string Sentence(Node root)
    {
        _ids.Clear();
        foreach (var d in root.Descendants())
            if (d.Id is not null) _ids[d.Id] = d;

        string body;
        var punct = ".";
        switch (root.Head)
        {
            case "define":
                body = Define(root);
                break;
            case "ynq":
                body = Clause(Only(root), new Ctx(Mood.Ynq));
                punct = "?";
                break;
            case "whq":
                body = Clause(Only(root), new Ctx(Mood.Whq));
                punct = "?";
                break;
            case "command":
                body = Clause(Only(root), new Ctx(Mood.Imp));
                break;
            case "assert":
                body = Clause(Only(root), new Ctx(Mood.Decl));
                break;
            default:
                throw new ArgumentException($"Unknown act '{root.Head}'");
        }
        body = Tidy(body);
        return body.Length == 0 ? body : char.ToUpperInvariant(body[0]) + body[1..] + punct;
    }

    static Node Only(Node root) => root.Positional.FirstOrDefault() ?? throw new ArgumentException("Act without clause");

    static string Tidy(string s)
    {
        var parts = s.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var joined = string.Join(' ', parts);
        return joined.Replace(" ,", ",").Replace(" n't", "n't").Replace(" 's", "'s").Replace(" 're", "'re")
            .Replace(" 'm", "'m").Replace(" 'll", "'ll").Replace(" 've", "'ve").Replace(" 'd", "'d").Replace(",,", ",");
    }

    static string Join(params string?[] parts) => string.Join(' ', parts.Where(p => !string.IsNullOrWhiteSpace(p)));

    static string Word(string head) => head.Replace('_', ' ');

    // ---------------------------------------------------------------- definitions

    string Define(Node def)
    {
        var term = def.Get("term")!;
        var body = def.Get("def")!;
        var marker = def.Atom("marker") ?? "means";
        var termText = term.IsEvent ? Infinitive(term, null) : NounPhrase(term, Case.Nom, null);
        var bodyText = body.IsEvent ? Infinitive(body, null) : NounPhrase(body, Case.Nom, null);
        var plural = term.IsMention && Agreement(term).Plural;
        var verb = marker switch
        {
            "is" => plural ? "are" : "is",
            "refers-to" => plural ? "refer to" : "refers to",
            _ => plural ? "mean" : "means",
        };
        return Join(termText, verb, bodyText);
    }

    // ---------------------------------------------------------------- clauses

    string Clause(Node n, Ctx ctx)
    {
        if (n.IsCoordination)
        {
            var parts = n.Positional.Select(c => Clause(c, ctx)).ToList();
            var conj = n.Head == "but" ? ", but" : n.Head;
            return string.Join($" {conj} ", parts).Replace(" , but ", ", but ");
        }
        if (n.Id is null && n.Head == "define") return Define(n);
        return Event(n, ctx);
    }

    string Event(Node e, Ctx ctx)
    {
        var main = MainClause(e, ctx);
        var front = new List<string>();
        var back = new List<string>();
        foreach (var p in e.Props)
        {
            if (p.Node is null || !Schema.Relations.Contains(p.Role)) continue;
            if (p.Role == "purpose")
            {
                back.Add(p.Node.Has("tense") || p.Node.Has("mod")
                    ? Join("so that", Clause(p.Node, new Ctx(Mood.Decl)))
                    : Infinitive(p.Node, null));
                continue;
            }
            var sub = Clause(p.Node, new Ctx(Mood.Decl));
            if (p.Role == "cause" && _o.CauseAsSo && ctx.Mood == Mood.Decl && ctx.GapId is null)
            {
                main = $"{sub}, so {main}";
                continue;
            }
            var marker = p.Role switch
            {
                "condition" => "if",
                "unless" => "unless",
                "cause" => _o.CauseAsSince ? "since" : "because",
                "concession" => _o.ConcessionAsThough ? "though" : "although",
                _ => p.Role,
            };
            if (_o.FrontSubordinate && ctx.GapId is null) front.Add($"{marker} {sub},");
            else back.Add($"{marker} {sub}");
        }
        return Join(Join(front.ToArray()), main, Join(back.ToArray()));
    }

    sealed record Agr(int Person, bool Plural);

    Agr Agreement(Node? subj)
    {
        if (subj is null) return new Agr(3, false);
        if (subj.IsCoordination)
        {
            var kids = subj.Positional.ToList();
            if (subj.Head == "and")
                return new Agr(kids.Any(k => k.Head == "I" && k.Has("pron")) ? 1 : kids.Any(k => k.Head == "you" && k.Has("pron")) ? 2 : 3, true);
            return Agreement(kids.LastOrDefault());
        }
        if (subj.Has("pron") && Words.Pronouns.TryGetValue(subj.Head, out var pr)) return new Agr(pr.Person, pr.Plural || pr.Person == 2);
        if (subj.Atom("num") == "pl") return new Agr(3, true);
        var det = subj.Atom("det");
        if (det is not null && Words.PluralDeterminers.Contains(det)) return new Agr(3, true);
        return new Agr(3, false);
    }

    /// <summary>Builds the verb group as a list of words, plus where the subject goes for inversion.</summary>
    List<string> VerbChain(Node e, Agr agr, bool finite, out bool doSupportUsed, bool needAux, bool imperative)
    {
        var chain = new List<(string Lemma, string Kind)>();
        var mod = e.Atom("mod");
        var tense = e.Atom("tense");
        var aspect = e.Atom("aspect");
        if (mod is not null) chain.Add((mod, "modal"));
        else if (tense == "fut") chain.Add(("will", "modal"));
        if (aspect is "perf" or "perfprog") chain.Add(("have", "perf"));
        if (aspect is "prog" or "perfprog") chain.Add(("be", "prog"));
        if (e.Atom("voice") == "pass") chain.Add(("be", "pass"));
        chain.Add((e.Head, "main"));

        doSupportUsed = false;
        if (finite && !imperative && needAux && chain.Count == 1 && e.Head != "be")
        {
            chain.Insert(0, ("do", "do"));
            doSupportUsed = true;
        }
        if (imperative && needAux)
        {
            chain.Insert(0, ("do", "do"));
            doSupportUsed = true;
        }

        var words = new List<string>();
        for (var i = 0; i < chain.Count; i++)
        {
            var (lemma, kind) = chain[i];
            string form;
            if (i == 0)
            {
                if (kind == "modal") form = lemma;
                else if (!finite || imperative) form = lemma;
                else form = Finite(lemma, tense == "past", agr);
            }
            else
            {
                var prev = chain[i - 1].Kind;
                form = prev switch
                {
                    "modal" or "do" => lemma,
                    "perf" => Morph.PastParticiple(lemma),
                    "prog" => Morph.PresentParticiple(lemma),
                    "pass" => Morph.PastParticiple(lemma),
                    _ => lemma,
                };
            }
            words.Add(Word(form));
        }
        return words;
    }

    static string Finite(string lemma, bool past, Agr agr)
    {
        if (lemma == "be") return Morph.Be(past, agr.Person, agr.Plural);
        if (past) return Morph.Past(lemma);
        if (agr.Person == 3 && !agr.Plural) return Morph.ThirdSingular(lemma);
        return lemma;
    }

    static string? NegContraction(string aux) => aux switch
    {
        "will" => "won't",
        "can" => "can't",
        "shall" => "shan't",
        "am" => null,
        "may" => null,
        "might" => null,
        _ => aux + "n't",
    };

    string MainClause(Node e, Ctx ctx)
    {
        var mood = ctx.Mood;
        var subjProp = e.Props.FirstOrDefault(p => p.Role == "subj");
        var subjNode = subjProp?.Node;
        var subjIsGap = subjProp?.Atom is not null && subjProp.Atom == ctx.GapId;
        var there = e.Has("there");
        var agrSource = subjIsGap && ctx.GapId is not null && _ids.TryGetValue(ctx.GapId, out var host) ? host : subjNode;
        var agr = Agreement(agrSource);

        var neg = e.Atom("pol") == "neg";
        var negWord = e.Atom("negword");
        var whArg = mood == Mood.Whq ? FindWhArgument(e) : null;
        var whAdv = e.Atom("wh-adv");
        var subjectIsWh = whArg is not null && whArg.Value.Role == "subj";
        var invert = mood == Mood.Ynq || (mood == Mood.Whq && !subjectIsWh);
        var imperative = mood == Mood.Imp;
        var infinitive = mood == Mood.Infinitive;
        var needAux = (neg && negWord is null) || invert;

        var chain = VerbChain(e, agr, finite: !infinitive, out _, needAux, imperative);

        var preAdverbs = e.Props.Where(p => p.Role == "adv" && p.Atom is not null && Words.PreverbalAdverbs.Contains(p.Atom)).Select(p => p.Atom!).ToList();
        var otherAdverbs = e.Props.Where(p => p.Role == "adv" && !(p.Atom is not null && Words.PreverbalAdverbs.Contains(p.Atom))).ToList();
        var please = imperative && otherAdverbs.RemoveAll(p => p.Atom == "please") > 0;
        // Short adverbs ("left", "well", "back", "quickly") follow the objects; time adverbs and adverb phrases end the clause.
        var midAdverbs = otherAdverbs.Where(p => p.Atom is not null && !TimeAdverbs.Contains(p.Atom)).Select(p => Word(p.Atom!)).ToList();
        var postAdverbs = otherAdverbs.Where(p => p.Node is not null || TimeAdverbs.Contains(p.Atom!))
            .Select(p => p.Node is not null ? Modifier(p.Node) : Word(p.Atom!)).ToList();
        if (negWord is not null) preAdverbs.Insert(0, negWord);

        // Where "not" and preverbal adverbs go: after the first auxiliary, or before a lone main verb.
        var lone = chain.Count == 1 && !(e.Head == "be" && !imperative && !infinitive);
        string negText = "";
        if (neg && negWord is null)
        {
            if (_o.Contractions && !infinitive && NegContraction(chain[0]) is { } contracted && !(imperative && false))
            {
                chain[0] = contracted;
            }
            else negText = "not";
        }

        // Subject text.
        string subjText;
        if (imperative) subjText = subjNode is not null && !subjNode.Has("implicit") ? NounPhrase(subjNode, Case.Nom, ctx.GapId) : "";
        else if (infinitive) subjText = "";
        else if (subjIsGap) subjText = "";
        else if (there) subjText = "there";
        else subjText = subjNode is not null ? NounPhrase(subjNode, Case.Nom, ctx.GapId) : "";

        if (_o.Contractions && mood == Mood.Decl && subjNode is not null && subjNode.Has("pron") && !there && !subjIsGap && chain.Count > 1 || (_o.Contractions && mood == Mood.Decl && subjNode is not null && subjNode.Has("pron") && e.Head == "be" && chain.Count == 1 && !there))
        {
            var c = SubjectContraction(chain[0]);
            if (c is not null && !(neg && negText == "" && chain[0].EndsWith("n't")))
            {
                subjText = subjText + c;
                chain[0] = "";
            }
        }

        // Complements.
        var comps = new List<string>();
        if (there && subjNode is not null) comps.Add(NounPhrase(subjNode, Case.Nom, ctx.GapId));
        var attr = e.Props.FirstOrDefault(p => p.Role == "attr");
        if (attr is not null && !(whArg is not null && ReferenceEquals(attr, whArg.Value.Prop))) comps.Add(attr.Node is { IsMention: true } ? NounPhrase(attr.Node, Case.Nom, ctx.GapId) : attr.Node is not null ? Modifier(attr.Node) : Word(attr.Atom!));
        var stranded = new List<string>();
        foreach (var role in new[] { "iobj", "obj" })
            foreach (var p in e.All(role))
            {
                if (whArg is not null && ReferenceEquals(p, whArg.Value.Prop)) continue;
                if (p.Atom is not null && p.Atom == ctx.GapId) continue;
                if (p.Node is not null) comps.Add(NounPhrase(p.Node, Case.Acc, ctx.GapId));
            }
        foreach (var p in e.Props.Where(p => Schema.IsOblique(p.Role)))
            if (whArg is not null && ReferenceEquals(p, whArg.Value.Prop) || p.Atom is not null && p.Atom == ctx.GapId)
                stranded.Add(Schema.Preposition(p.Role));
        comps.AddRange(stranded);
        comps.AddRange(midAdverbs);
        foreach (var p in e.All("xcomp"))
            if (p.Node is not null) comps.Add(Infinitive(p.Node, ctx.GapId));
        foreach (var p in e.Props.Where(p => Schema.IsOblique(p.Role)))
            if (p.Node is not null && !(whArg is not null && ReferenceEquals(p, whArg.Value.Prop)))
                comps.Add(Join(Schema.Preposition(p.Role), NounPhrase(p.Node, Case.Acc, ctx.GapId)));
        foreach (var p in e.All("comp"))
            if (p.Node is not null)
                comps.Add(p.Node.Head == "define" ? Join(_o.DropComplementizer ? null : "that", Define(p.Node))
                    : Join(_o.DropComplementizer ? null : "that", Clause(p.Node, new Ctx(Mood.Decl))));
        comps.AddRange(postAdverbs);

        var pre = string.Join(' ', preAdverbs);
        if (infinitive)
        {
            var verbs = string.Join(' ', chain);
            return Join(negText, "to", pre, verbs, Join(comps.ToArray()));
        }
        if (imperative)
        {
            if (please) return Join("please", ImperativeBody());
            return ImperativeBody();
        }

        string ImperativeBody()
        {
            if (chain.Count > 1 && (chain[0] == "do" || chain[0] == "don't"))
                return Join(chain[0], negText, subjText, pre, string.Join(' ', chain.Skip(1)), Join(comps.ToArray()));
            return Join(subjText, pre, string.Join(' ', chain), Join(comps.ToArray()));
        }

        string verbGroup;
        if (lone) verbGroup = Join(pre, chain[0]);
        else verbGroup = Join(chain[0], negText, pre, string.Join(' ', chain.Skip(1)));
        if (lone && negText != "") verbGroup = Join(negText, verbGroup); // unreachable for lexical verbs; kept for safety

        if (!invert)
        {
            var whFront = "";
            if (subjectIsWh) subjText = NounPhrase(whArg!.Value.Node, Case.Nom, ctx.GapId);
            var relPron = ctx.GapId is not null ? RelativePronoun(ctx.GapId) : null;
            if (subjIsGap) return Join(relPron, verbGroup, Join(comps.ToArray()));
            if (relPron is not null) whFront = relPron;
            return Join(whFront, subjText, verbGroup, Join(comps.ToArray()));
        }

        // Inverted: first verb word before the subject. A contracted negation travels with it ("Doesn't she...").
        var first = chain[0];
        var rest = string.Join(' ', chain.Skip(1));
        string front = "";
        if (mood == Mood.Whq)
            front = whArg is not null ? NounPhrase(whArg.Value.Node, Case.Acc, ctx.GapId) : whAdv ?? "";
        return Join(front, first, subjText, negText, pre, rest, Join(comps.ToArray()));
    }

    static readonly HashSet<string> TimeAdverbs = ["yesterday", "today", "tomorrow", "tonight", "now", "later", "then", "soon", "again", "early", "late", "already", "recently", "ago", "lately"];

    static string? SubjectContraction(string aux) => aux switch
    {
        "am" => "'m",
        "is" or "has" => "'s",
        "are" => "'re",
        "will" => "'ll",
        "would" or "had" => "'d",
        "have" => "'ve",
        _ => null,
    };

    record struct WhArg(Prop Prop, string Role, Node Node);

    static WhArg? FindWhArgument(Node e)
    {
        foreach (var p in e.Props)
        {
            if (p.Node is null) continue;
            if ((Schema.IsArgumentRole(p.Role) || p.Role == "attr") && p.Node.Has("wh")) return new WhArg(p, p.Role, p.Node);
        }
        return null;
    }

    string Infinitive(Node e, string? gap) => MainClause(e, new Ctx(Mood.Infinitive, gap));

    string RelativePronoun(string hostId)
    {
        if (!_ids.TryGetValue(hostId, out var host)) return "that";
        if (_o.RelativeWho && (host.Has("propn") || host.Has("pron"))) return "who";
        return "that";
    }

    // ---------------------------------------------------------------- noun phrases

    string NounPhrase(Node m, Case c, string? gap)
    {
        if (m.IsCoordination)
        {
            var kids = m.Positional.Select(k => NounPhrase(k, c, gap)).ToList();
            if (kids.Count == 2) return $"{kids[0]} {m.Head} {kids[1]}";
            return string.Join(", ", kids.Take(kids.Count - 1)) + $", {m.Head} " + kids[^1];
        }

        if (m.Has("pron") && Words.Pronouns.TryGetValue(m.Head, out var pr))
        {
            if (m.Has("refl")) return pr.Refl;
            return c switch { Case.Nom => pr.Nom, Case.Acc => pr.Acc, _ => pr.PossDet };
        }
        if (m.Has("wh") && !m.Has("det") && m.Head is "who" or "what")
            return c == Case.Poss ? "whose" : m.Head;

        var plural = m.Atom("num") == "pl";
        var noun = Word(plural ? Morph.Plural(m.Head) : m.Head);
        var adjs = m.Props.Where(p => p.Role == "adj").Select(p => p.Node is not null ? Modifier(p.Node) : Word(p.Atom!)).ToList();
        var post = new List<string>();
        foreach (var p in m.Props.Where(p => Schema.IsOblique(p.Role)))
            if (p.Node is not null) post.Add(Join(Schema.Preposition(p.Role), NounPhrase(p.Node, Case.Acc, gap)));
        foreach (var p in m.All("rel"))
            if (p.Node is not null) post.Add(Clause(p.Node, new Ctx(Mood.Decl, m.Id)));

        string? det = null;
        var poss = m.Get("poss");
        if (poss is not null) det = Possessive(poss);
        else if (m.Atom("det") is { } d)
        {
            var next = adjs.Count > 0 ? adjs[0] : noun;
            det = d is "a" or "an" ? Morph.IndefiniteArticle(next) : Word(d);
        }

        var compounds = m.Atoms("compound").Select(Word).ToArray();
        if (det is not null && m.Atom("det") is "a" or "an" && adjs.Count == 0 && compounds.Length > 0) det = Morph.IndefiniteArticle(compounds[0]);
        var core = Join(det, Join(adjs.ToArray()), Join(compounds), noun, Join(post.ToArray()));
        return c == Case.Poss ? core + "'s" : core;
    }

    string Possessive(Node poss)
    {
        if (poss.Has("pron") && Words.Pronouns.TryGetValue(poss.Head, out var pr)) return pr.PossDet;
        if (poss.Has("wh") && poss.Head == "who") return "whose";
        var np = NounPhrase(poss, Case.Nom, null);
        return np.EndsWith('s') && poss.Atom("num") == "pl" ? np + "'" : np + "'s";
    }

    static string Modifier(Node n)
    {
        var advs = n.All("adv").Select(p => p.Node is not null ? Modifier(p.Node) : Word(p.Atom!));
        return Join(Join(advs.ToArray()), Word(n.Head));
    }
}
