namespace Language.Core.Ir;

/// <summary>Checks that an IR tree is well formed according to docs/IR-v0.md.
/// Validation is structural only. It never judges whether content is plausible.</summary>
public static class Validator
{
    public static List<string> Validate(Node root)
    {
        var errors = new List<string>();
        if (!Schema.Acts.Contains(root.Head))
        {
            errors.Add($"Root head must be one of {string.Join("/", Schema.Acts)}, got '{root.Head}'");
            return errors;
        }

        var ids = new Dictionary<string, Node>();
        foreach (var d in root.Descendants())
        {
            if (d.Id is null) continue;
            if (!ids.TryAdd(d.Id, d)) errors.Add($"Duplicate id {d.Id}");
        }

        if (root.Head == "define")
            CheckDefine(root, errors, ids);
        else
        {
            var kids = root.Positional.ToList();
            if (kids.Count != 1) errors.Add($"Act '{root.Head}' must have exactly one clause, found {kids.Count}");
            else CheckClause(kids[0], errors, ids, root.Head == "command" ? Finiteness.Imperative : Finiteness.Finite);
        }

        foreach (var p in root.Props.Where(p => p.Role != ""))
            if (p.Role == "ambig") CheckEnum(p, Schema.AmbiguityTypes, errors);
            else if (!(root.Head == "define" && p.Role is "term" or "def" or "marker"))
                errors.Add($"Unexpected role :{p.Role} on act node");

        if (root.Head == "whq" && !root.Descendants().Any(d => d.Has("wh")) && !root.Descendants().Any(d => d.Has("wh-adv")))
            errors.Add("whq needs a :wh mention or a :wh-adv");

        return errors;
    }

    enum Finiteness { Finite, Imperative, NonFinite }

    static void CheckDefine(Node def, List<string> errors, Dictionary<string, Node> ids)
    {
        var term = def.Get("term");
        var body = def.Get("def");
        if (term is null) errors.Add("define needs :term");
        if (body is null) errors.Add("define needs :def");
        var marker = def.Props.FirstOrDefault(p => p.Role == "marker");
        if (marker is null) errors.Add("define needs :marker");
        else CheckEnum(marker, Schema.DefineMarkers, errors);
        foreach (var part in new[] { term, body })
        {
            if (part is null) continue;
            if (part.IsEvent) CheckClause(part, errors, ids, Finiteness.NonFinite);
            else if (part.IsMention) CheckMention(part, errors, ids);
            else errors.Add($"define :term/:def must be a mention or event, got ({part.Head})");
        }
    }

    static void CheckClause(Node n, List<string> errors, Dictionary<string, Node> ids, Finiteness fin)
    {
        if (Schema.Coordinators.Contains(n.Head) && n.Id is null)
        {
            var kids = n.Positional.ToList();
            if (kids.Count < 2) errors.Add($"Coordination ({n.Head}) needs at least two members");
            foreach (var k in kids) CheckClause(k, errors, ids, fin);
            return;
        }
        if (n.Head == "define" && n.Id is null)
        {
            CheckDefine(n, errors, ids);
            return;
        }
        if (!n.IsEvent)
        {
            errors.Add($"Expected an event (eN ...) but found ({n.Id} {n.Head})");
            return;
        }
        CheckEvent(n, errors, ids, fin);
    }

    static void CheckEvent(Node e, List<string> errors, Dictionary<string, Node> ids, Finiteness fin)
    {
        var hasTense = e.Has("tense");
        if (fin == Finiteness.Finite && !hasTense && !e.Has("mod"))
            errors.Add($"{e.Id}: finite clause needs :tense (or :mod)");
        if (fin != Finiteness.Finite && hasTense)
            errors.Add($"{e.Id}: non-finite or imperative clause must not have :tense");

        foreach (var p in e.Props)
        {
            switch (p.Role)
            {
                case "tense": CheckEnum(p, Schema.Tenses, errors); break;
                case "aspect": CheckEnum(p, Schema.Aspects, errors); break;
                case "pol": CheckEnum(p, ["neg"], errors); break;
                case "negword": CheckEnum(p, Schema.NegWords, errors); break;
                case "mod": CheckEnum(p, Schema.Modals, errors); break;
                case "voice": CheckEnum(p, ["pass"], errors); break;
                case "wh-adv": CheckEnum(p, Schema.WhAdverbs, errors); break;
                case "there": if (!p.IsFlag) errors.Add($"{e.Id}: :there is a flag"); break;
                case "subj" or "obj" or "iobj":
                case var r when Schema.IsOblique(r):
                    CheckArgument(e, p, errors, ids);
                    break;
                case "attr":
                    if (p.Node is { IsMention: true }) CheckMention(p.Node, errors, ids);
                    else if (p.Node is { Id: not null }) errors.Add($"{e.Id}: :attr must be a mention or adjective phrase");
                    else if (p.Node is not null) CheckModifier(p.Node, errors);
                    break;
                case "xcomp" or "purpose":
                    if (p.Node is null) errors.Add($"{e.Id}: :{p.Role} needs a clause");
                    else CheckClause(p.Node, errors, ids, p.Role == "purpose" && (p.Node.Has("tense") || p.Node.Has("mod")) ? Finiteness.Finite : Finiteness.NonFinite);
                    break;
                case "comp":
                    if (p.Node is null) errors.Add($"{e.Id}: :comp needs a clause");
                    else CheckClause(p.Node, errors, ids, Finiteness.Finite);
                    break;
                case "adv":
                    if (p.Node is not null) CheckModifier(p.Node, errors);
                    else if (p.Atom is null) errors.Add($"{e.Id}: :adv needs a value");
                    break;
                case var r when Schema.Relations.Contains(r):
                    if (p.Node is null) errors.Add($"{e.Id}: :{r} needs a clause");
                    else CheckClause(p.Node, errors, ids, Finiteness.Finite);
                    break;
                case "unk" or "lex": break;
                default:
                    errors.Add($"{e.Id}: unexpected role :{p.Role} on event");
                    break;
            }
        }
        if (e.Has("negword") && !e.Has("pol")) errors.Add($"{e.Id}: :negword requires :pol neg");
    }

    static void CheckArgument(Node e, Prop p, List<string> errors, Dictionary<string, Node> ids)
    {
        if (p.Node is not null)
        {
            if (p.Node.IsCoordination && (p.Node.Id is null || p.Node.IsMention))
            {
                var kids = p.Node.Positional.ToList();
                if (kids.Count < 2) errors.Add($"{e.Id}: coordinated :{p.Role} needs two members");
                foreach (var k in kids)
                    if (k.IsMention) CheckMention(k, errors, ids);
                    else errors.Add($"{e.Id}: coordinated :{p.Role} members must be mentions");
            }
            else if (p.Node.IsMention) CheckMention(p.Node, errors, ids);
            else errors.Add($"{e.Id}: :{p.Role} must be a mention");
        }
        else if (p.Atom is not null)
        {
            // An atom argument is a relative-clause gap: a reference to the mention the clause modifies.
            if (!ids.TryGetValue(p.Atom, out var target) || !target.IsMention)
                errors.Add($"{e.Id}: :{p.Role} refers to unknown mention '{p.Atom}'");
            else if (!target.All("rel").Any(r => r.Node is not null && r.Node.Descendants().Contains(e)))
                errors.Add($"{e.Id}: :{p.Role} {p.Atom} is only allowed as a gap inside {p.Atom}'s :rel clause");
        }
        else errors.Add($"{e.Id}: :{p.Role} needs a value");
    }

    static void CheckMention(Node m, List<string> errors, Dictionary<string, Node> ids)
    {
        foreach (var p in m.Props)
        {
            switch (p.Role)
            {
                case "det": if (p.Atom is null) errors.Add($"{m.Id}: :det needs a word"); break;
                case "num": CheckEnum(p, ["pl"], errors); break;
                case "poss":
                    if (p.Node is { IsMention: true }) CheckMention(p.Node, errors, ids);
                    else errors.Add($"{m.Id}: :poss must be a mention");
                    break;
                case "adj":
                    if (p.Node is not null) CheckModifier(p.Node, errors);
                    else if (p.Atom is null) errors.Add($"{m.Id}: :adj needs a value");
                    break;
                case "compound":
                    if (p.Atom is null) errors.Add($"{m.Id}: :compound needs a noun (atom)");
                    break;
                case "propn" or "pron" or "wh" or "refl" or "implicit":
                    if (!p.IsFlag) errors.Add($"{m.Id}: :{p.Role} is a flag");
                    break;
                case "ref":
                    if (p.Atom is null) errors.Add($"{m.Id}: :ref needs a mention id or 'none'");
                    else if (p.Atom != "none" && (!ids.TryGetValue(p.Atom, out var t) || !t.IsMention))
                        errors.Add($"{m.Id}: :ref points to unknown mention '{p.Atom}'");
                    else if (p.Atom == m.Id) errors.Add($"{m.Id}: :ref points to itself");
                    break;
                case var r when Schema.IsOblique(r):
                    if (p.Node is { IsMention: true }) CheckMention(p.Node, errors, ids);
                    else errors.Add($"{m.Id}: :{r} must be a mention");
                    break;
                case "rel":
                    if (p.Node is not { IsEvent: true }) errors.Add($"{m.Id}: :rel must be an event");
                    else
                    {
                        CheckEvent(p.Node, errors, ids, Finiteness.Finite);
                        if (!p.Node.Descendants().Any(d => d.Props.Any(q => q.Atom == m.Id && Schema.IsArgumentRole(q.Role))))
                            errors.Add($"{m.Id}: relative clause {p.Node.Id} has no gap referring to {m.Id}");
                    }
                    break;
                case "unk" or "lex": break;
                default:
                    errors.Add($"{m.Id}: unexpected role :{p.Role} on mention");
                    break;
            }
        }
        if (m.Has("ref") && !m.Has("pron")) errors.Add($"{m.Id}: only pronouns (:pron) carry :ref");
        if (m.Has("refl") && !m.Has("pron")) errors.Add($"{m.Id}: :refl requires :pron");
    }

    static void CheckModifier(Node n, List<string> errors)
    {
        if (n.Id is not null) errors.Add($"Modifier phrase ({n.Id} {n.Head}) must not have an id");
        foreach (var p in n.Props)
            if (p.Role != "adv") errors.Add($"Modifier phrase ({n.Head}) only allows :adv, found :{p.Role}");
            else if (p.Node is not null) CheckModifier(p.Node, errors);
    }

    static void CheckEnum(Prop p, string[] allowed, List<string> errors)
    {
        if (p.Atom is null || !allowed.Contains(p.Atom))
            errors.Add($":{p.Role} must be one of {string.Join("/", allowed)}, got '{p.Atom ?? (p.Node is null ? "(flag)" : "(node)")}'");
    }
}
