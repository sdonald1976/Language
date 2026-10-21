using Language.Core.Text;

namespace Language.Core.Ir;

/// <summary>
/// Aligns IR content to the input text. Every event, mention and modifier head, and every lexical atom
/// (:adj, :adv, :attr, :det, preposition of :obl-*), must be realizable as some input token,
/// unless the node is marked :implicit. Anything that cannot be aligned was invented by the reader.
/// </summary>
public sealed class Anchoring
{
    public Dictionary<Node, int> NodeAnchors { get; } = new(ReferenceEqualityComparer.Instance);
    public List<string> Unanchored { get; } = new();
    public int ContentItems { get; private set; }

    public static readonly string[] LexicalAtomRoles = ["adj", "adv", "attr", "det", "compound"];

    public static Anchoring Compute(Node root, IReadOnlyList<Token> tokens)
    {
        var a = new Anchoring();
        var used = new HashSet<int>();
        foreach (var (node, path, parent, role) in root.Walk())
        {
            if (parent is null && Schema.Acts.Contains(node.Head)) continue;
            if ((node.IsCoordination || node.Head == "define"))
            {
                a.ContentItems++;
                if (node.Head != "define" && Find(node.Head, tokens, used, consume: false) < 0) a.Unanchored.Add($"{path}: coordinator '{node.Head}'");
                continue;
            }
            if (node.Has("implicit")) continue;
            a.ContentItems++;
            var idx = Find(node.Head, tokens, used, consume: true);
            if (idx >= 0) a.NodeAnchors[node] = idx;
            else a.Unanchored.Add($"{path}: '{node.Head}'");

            foreach (var p in node.Props)
            {
                if (p.Atom is null) continue;
                if (LexicalAtomRoles.Contains(p.Role) || p.Role == "negword")
                {
                    a.ContentItems++;
                    if (Find(p.Atom, tokens, used, consume: false) < 0) a.Unanchored.Add($"{path}/:{p.Role} '{p.Atom}'");
                }
            }
            foreach (var p in node.Props.Where(p => Schema.IsOblique(p.Role)))
            {
                a.ContentItems++;
                if (Find(p.Role[4..], tokens, used, consume: false) < 0) a.Unanchored.Add($"{path}/:{p.Role} (preposition)");
            }
        }
        return a;
    }

    /// <summary>Finds the first token sequence that can realize the (possibly multi-word, '_'-joined) lemma.</summary>
    static int Find(string lemma, IReadOnlyList<Token> tokens, HashSet<int> used, bool consume)
    {
        var parts = lemma.Split('_', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0) return -1;
        int fallback = -1;
        for (var i = 0; i + parts.Length <= tokens.Count; i++)
        {
            var ok = true;
            for (var j = 0; j < parts.Length && ok; j++)
                ok = j == parts.Length - 1
                    ? Morph.CanSurfaceAs(parts[j], tokens[i + j].Text)
                    : string.Equals(parts[j], tokens[i + j].Text, StringComparison.OrdinalIgnoreCase);
            if (!ok) continue;
            if (!consume || !used.Contains(i))
            {
                if (consume) used.Add(i);
                return i;
            }
            if (fallback < 0) fallback = i;
        }
        return fallback;
    }

    public double InventedRate => ContentItems == 0 ? 0 : (double)Unanchored.Count / ContentItems;
}
