namespace Language.Core.Ir;

/// <summary>The fixed vocabulary of IR v0: heads with special meaning, roles, and allowed feature values.</summary>
public static class Schema
{
    public const string Version = "0.1";

    public static readonly string[] Acts = ["assert", "ynq", "whq", "command", "define"];
    public static readonly string[] Coordinators = ["and", "or", "but"];

    public static readonly string[] Tenses = ["past", "pres", "fut"];
    public static readonly string[] Aspects = ["prog", "perf", "perfprog"];
    public static readonly string[] Modals = ["can", "could", "may", "might", "must", "should", "would", "shall"];
    public static readonly string[] NegWords = ["never"];
    public static readonly string[] WhAdverbs = ["where", "when", "why", "how"];
    public static readonly string[] DefineMarkers = ["means", "is", "refers-to"];
    public static readonly string[] AmbiguityTypes = ["attachment", "scope", "lexical", "structural"];

    /// <summary>Clause-to-clause relations. Each is a role on the main event whose value is the subordinate event.</summary>
    public static readonly string[] Relations =
        ["condition", "unless", "cause", "concession", "when", "before", "after", "while", "until", "purpose"];

    public static readonly string[] EventFeatures = ["tense", "aspect", "pol", "negword", "mod", "voice", "there"];
    public static readonly string[] EventArgs = ["subj", "obj", "iobj", "attr", "xcomp", "comp"];
    public static readonly string[] MentionFeatures = ["det", "num", "poss", "adj", "compound", "propn", "pron", "wh", "refl", "implicit", "ref"];

    /// <summary>Roles added by the lexicon layer at runtime. Not produced by the model and ignored by metrics.</summary>
    public static readonly string[] RuntimeRoles = ["unk", "lex"];

    static readonly string[] Order =
    [
        "tense", "aspect", "pol", "negword", "mod", "voice", "there",
        "det", "num", "poss", "adj", "compound", "propn", "pron", "wh", "refl", "implicit", "ref",
        "term", "def", "marker",
        "subj", "obj", "iobj", "attr", "obl-*", "xcomp", "comp", "wh-adv", "adv", "rel",
        "condition", "unless", "cause", "concession", "when", "before", "after", "while", "until", "purpose",
        "ambig", "", "unk", "lex",
    ];

    public static int Rank(string role)
    {
        var key = role.StartsWith("obl-") ? "obl-*" : role;
        var i = Array.IndexOf(Order, key);
        return i < 0 ? Order.Length : i;
    }

    public static bool IsOblique(string role) => role.StartsWith("obl-") && role.Length > 4;
    public static string Preposition(string oblRole) => oblRole[4..].Replace('_', ' ');

    public static bool IsArgumentRole(string role) => role is "subj" or "obj" or "iobj" || IsOblique(role);

    /// <summary>Sorts every node's properties into canonical order (stable), recursively.
    /// Training targets are always canonical so the model never has to learn arbitrary orderings.</summary>
    public static Node Canonicalize(Node node)
    {
        var sorted = node.Props
            .Select((p, i) => (p, i))
            .OrderBy(x => Rank(x.p.Role))
            .ThenBy(x => x.i)
            .Select(x => x.p)
            .ToList();
        node.Props.Clear();
        node.Props.AddRange(sorted);
        foreach (var c in node.Children) Canonicalize(c);
        return node;
    }

    /// <summary>Removes runtime-only annotations (lexicon marks) so trees can be compared with model output.</summary>
    public static Node StripRuntime(Node node)
    {
        var n = node.Clone();
        foreach (var d in n.Descendants())
            d.Props.RemoveAll(p => RuntimeRoles.Contains(p.Role));
        return n;
    }

    /// <summary>Renumbers ids e1.., m1.. in depth-first order and rewrites references to match.</summary>
    public static Node Renumber(Node root)
    {
        var n = root.Clone();
        var map = new Dictionary<string, string>();
        int e = 0, m = 0;
        foreach (var d in n.Descendants())
        {
            if (d.Id is null) continue;
            var fresh = d.IsEvent ? $"e{++e}" : $"m{++m}";
            map[d.Id] = fresh;
            d.Id = fresh;
        }
        foreach (var d in n.Descendants())
            for (var i = 0; i < d.Props.Count; i++)
            {
                var p = d.Props[i];
                if (p.Atom is not null && map.TryGetValue(p.Atom, out var r) && Node.IdPattern.IsMatch(p.Atom))
                    d.Props[i] = p with { Atom = r };
            }
        return n;
    }
}
