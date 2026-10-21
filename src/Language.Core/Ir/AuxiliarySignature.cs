using Language.Core.Realize;
using Language.Core.Text;

namespace Language.Core.Ir;

/// <summary>
/// A model-independent check of the grammatical features a writer must preserve: tense, aspect, voice, polarity
/// and modality all surface as auxiliaries and negators. A faithful sentence has the same multiset of these words as the
/// rule realizer's rendering of the same IR, up to contractions. This catches "was sleeping" for a simple past
/// even when the model's own re-parse makes the same mistake.
/// </summary>
public static class AuxiliarySignature
{
    static readonly HashSet<string> Aux =
    [
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must", "not", "never",
    ];

    /// <summary>Contractions and their possible expansions ('s and 'd are ambiguous).</summary>
    static readonly Dictionary<string, string[]> Clitics = new()
    {
        ["n't"] = ["not"], ["'m"] = ["am"], ["'re"] = ["are"], ["'ve"] = ["have"], ["'ll"] = ["will"],
        ["'s"] = ["is", "has"], ["'d"] = ["would", "had"], ["wo"] = ["will"], ["ca"] = ["can"], ["sha"] = ["shall"],
    };

    /// <summary>Null when the text's auxiliaries match the IR, otherwise a short explanation.</summary>
    public static string? Mismatch(Node ir, string text)
    {
        // The reference rendering uses no contractions, so any 's in it is possessive.
        var expected = Signature(Realizer.Realize(ir), out _);
        var actual = Signature(text, out var ambiguous);
        foreach (var options in ambiguous)
        {
            var hit = options.FirstOrDefault(o => expected.GetValueOrDefault(o) > actual.GetValueOrDefault(o));
            if (hit is null && options.Contains("is")) continue; // possessive 's
            var pick = hit ?? options[0];
            actual[pick] = actual.GetValueOrDefault(pick) + 1;
        }
        // "never" and "not" may both appear in some realizations ("did not ever"); compare negation as one class.
        Merge(expected, "never", "not");
        Merge(actual, "never", "not");
        var diffs = expected.Keys.Union(actual.Keys)
            .Where(k => expected.GetValueOrDefault(k) != actual.GetValueOrDefault(k))
            .Select(k => $"{k}: expected {expected.GetValueOrDefault(k)}, found {actual.GetValueOrDefault(k)}")
            .ToList();
        return diffs.Count == 0 ? null : "auxiliaries differ (" + string.Join("; ", diffs) + ")";
    }

    static void Merge(Dictionary<string, int> d, string from, string into)
    {
        if (!d.TryGetValue(from, out var n)) return;
        d.Remove(from);
        d[into] = d.GetValueOrDefault(into) + n;
    }

    static Dictionary<string, int> Signature(string text, out List<string[]> ambiguous)
    {
        var counts = new Dictionary<string, int>();
        ambiguous = new List<string[]>();
        var tokens = Tokenizer.Tokenize(text);
        for (var i = 0; i < tokens.Count; i++)
        {
            var t = tokens[i].Lower;
            // "do/does/did" is only an auxiliary when a verb follows (not "did the dishes"); keep it: realizer and model both use do-support identically.
            if (Clitics.TryGetValue(t, out var exp))
            {
                if (t is "wo" or "ca" or "sha" && !(i + 1 < tokens.Count && tokens[i + 1].Lower == "n't")) continue;
                if (exp.Length == 1) counts[exp[0]] = counts.GetValueOrDefault(exp[0]) + 1;
                else ambiguous.Add(exp);
                continue;
            }
            if (Aux.Contains(t)) counts[t] = counts.GetValueOrDefault(t) + 1;
        }
        return counts;
    }
}
