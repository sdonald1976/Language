using Language.Core.Text;

namespace Language.Core.Ir;

/// <summary>The converse of <see cref="Anchoring"/>: content words of a text that no part of the IR licenses.
/// Used to catch omissions (a writer dropping content, or a converter dropping words).</summary>
public static class Coverage
{
    public static IEnumerable<string> Uncovered(Node ir, IReadOnlyList<Token> tokens)
    {
        var allowed = Licensed(ir);
        foreach (var t in tokens)
        {
            if (Words.IsPunctuation(t.Text) || Words.FunctionWords.Contains(t.Text) || t.Text.All(char.IsDigit)) continue;
            if (!allowed.Any(a => Morph.CanSurfaceAs(a, t.Text))) yield return t.Text;
        }
    }

    /// <summary>Every lemma the IR licenses: heads and lexical atoms, prepositions, modals, negators, relation markers.</summary>
    public static HashSet<string> Licensed(Node ir)
    {
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "not", "never", "will" };
        foreach (var d in ir.Descendants())
        {
            foreach (var part in d.Head.Split('_')) set.Add(part);
            if (d.Head == "define") set.UnionWith(["mean", "is", "refer"]);
            foreach (var p in d.Props)
            {
                if (Schema.IsOblique(p.Role)) foreach (var w in p.Role[4..].Split('_')) set.Add(w);
                if (p.Atom is not null && !Node.IdPattern.IsMatch(p.Atom))
                    foreach (var w in p.Atom.Split('_'))
                    {
                        set.Add(w);
                        set.UnionWith(w.Split('-')); // "refers-to"
                    }
                if (Schema.Relations.Contains(p.Role)) set.UnionWith(["if", "unless", "because", "since", "so", "although", "though", "even", "order"]);
            }
        }
        return set;
    }
}
