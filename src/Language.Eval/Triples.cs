using Language.Core.Ir;
using Language.Core.Text;

namespace Language.Eval;

/// <summary>
/// Converts an IR tree to a multiset of (relation, source, target) triples whose node names are
/// <b>anchors</b> (token positions) rather than ids. Two trees for the same sentence can be compared
/// without a Smatch-style search, because nodes are aligned by the words they stand for.
/// </summary>
public sealed class TripleSet
{
    public List<string> Triples { get; } = new();
    public Dictionary<Node, string> Keys { get; } = new(ReferenceEqualityComparer.Instance);

    public static TripleSet Build(Node root, IReadOnlyList<Token> tokens)
    {
        var stripped = Schema.StripRuntime(root);
        var anchoring = Anchoring.Compute(stripped, tokens);
        var set = new TripleSet();
        var byId = new Dictionary<string, string>();

        foreach (var (node, path, parent, _) in stripped.Walk())
        {
            string key;
            if (parent is null) key = "root";
            else if (anchoring.NodeAnchors.TryGetValue(node, out var idx)) key = $"t{idx}";
            else key = "p" + path; // implicit, unanchored, coordinators and define nodes: identified by their role path
            set.Keys[node] = key;
            if (node.Id is not null) byId[node.Id] = key;
        }

        foreach (var (node, _, _, _) in stripped.Walk())
        {
            var k = set.Keys[node];
            set.Triples.Add($"instance|{k}|{node.Head.ToLowerInvariant()}");
            foreach (var p in node.Props)
            {
                var role = p.Role == "" ? "arg" : p.Role;
                if (p.Node is not null) set.Triples.Add($"{role}|{k}|{set.Keys[p.Node]}");
                else if (p.Atom is null) set.Triples.Add($"{role}|{k}|+");
                else if (Node.IdPattern.IsMatch(p.Atom) && byId.TryGetValue(p.Atom, out var target)) set.Triples.Add($"{role}|{k}|@{target}");
                else set.Triples.Add($"{role}|{k}|{p.Atom.ToLowerInvariant()}");
            }
        }
        return set;
    }

    /// <summary>Multiset intersection size.</summary>
    public static int Matches(IEnumerable<string> a, IEnumerable<string> b)
    {
        var counts = a.GroupBy(x => x).ToDictionary(g => g.Key, g => g.Count());
        var m = 0;
        foreach (var t in b)
            if (counts.TryGetValue(t, out var c) && c > 0) { counts[t] = c - 1; m++; }
        return m;
    }
}

/// <summary>Running precision/recall/F1 counts.</summary>
public sealed class Prf
{
    public int Matched, Predicted, Gold;

    public void Add(int matched, int predicted, int gold)
    {
        Matched += matched;
        Predicted += predicted;
        Gold += gold;
    }

    public double Precision => Predicted == 0 ? 0 : (double)Matched / Predicted;
    public double Recall => Gold == 0 ? 0 : (double)Matched / Gold;
    public double F1 => Precision + Recall == 0 ? 0 : 2 * Precision * Recall / (Precision + Recall);
}

/// <summary>Running accuracy.</summary>
public sealed class Acc
{
    public int Correct, Total;
    public void Add(bool ok) { Total++; if (ok) Correct++; }
    public double Value => Total == 0 ? double.NaN : (double)Correct / Total;
}
