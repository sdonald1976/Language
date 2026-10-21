using Language.Core.Data;
using Language.Core.Ir;
using Language.Core.Text;

namespace Language.Eval;

/// <summary>Writing metrics (DESIGN.md §6.2). Additions and omissions are judged against the IR lexically;
/// round-trip fidelity uses a re-parse of the output (when supplied).</summary>
public sealed class WriteScores
{
    public int Items;
    public Acc HasAddition = new(), HasForbidden = new(), HasOmission = new(), ActPreserved = new(), Produced = new(), AuxPreserved = new();
    public Prf RoundTrip = new();
    public int AddedTokens, OutputContentTokens;

    public Dictionary<string, object> Summary() => new()
    {
        ["items"] = Items,
        ["produced"] = Produced.Value,
        ["unsupported_addition_rate"] = HasAddition.Value,
        ["forbidden_word_rate"] = HasForbidden.Value,
        ["omission_rate"] = HasOmission.Value,
        ["sentence_type_preserved"] = ActPreserved.Value,
        ["tense_aspect_polarity_preserved"] = AuxPreserved.Value,
        ["added_token_share"] = OutputContentTokens == 0 ? 0 : (double)AddedTokens / OutputContentTokens,
        ["round_trip_triple_f1"] = RoundTrip.Gold == 0 ? double.NaN : RoundTrip.F1,
    };
}

public sealed record WriteItemResult(string Id, string? Output, List<string> Additions, List<string> Omissions, List<string> Forbidden, double? RoundTripF1, string? AuxMismatch = null);

public static class WriteEvaluator
{
    /// <summary>Words a writer may always add: grammar, not content.</summary>
    static readonly HashSet<string> Free = new(Words.FunctionWords, StringComparer.OrdinalIgnoreCase);

    public static (Dictionary<string, WriteScores> Slices, List<WriteItemResult> Items) Evaluate(
        IEnumerable<Item> gold, IReadOnlyDictionary<string, string?> outputs, IReadOnlyDictionary<string, string?>? reparsed = null)
    {
        var slices = new Dictionary<string, WriteScores>();
        WriteScores Slice(string n) => slices.TryGetValue(n, out var s) ? s : slices[n] = new WriteScores();
        var results = new List<WriteItemResult>();

        foreach (var item in gold)
        {
            if (item.Ir is null) continue;
            var ir = Linear.Parse(item.Ir);
            var output = outputs.GetValueOrDefault(item.Id);
            var targets = new List<WriteScores> { Slice("all"), Slice(item.Category ?? "uncategorized") };
            foreach (var t in item.Tags ?? []) targets.Add(Slice($"{item.Category}:{t}"));

            var additions = new List<string>();
            var omissions = new List<string>();
            var forbidden = new List<string>();
            double? rtF1 = null;
            var contentTokens = 0;
            if (!string.IsNullOrWhiteSpace(output))
            {
                var tokens = Tokenizer.Tokenize(output);
                var allowed = AllowedLemmas(ir);
                foreach (var t in tokens)
                {
                    if (Words.IsPunctuation(t.Text) || Free.Contains(t.Text) || t.Text.All(char.IsDigit)) continue;
                    contentTokens++;
                    if (!allowed.Any(a => Morph.CanSurfaceAs(a, t.Text))) additions.Add(t.Text);
                }
                omissions.AddRange(Anchoring.Compute(Schema.StripRuntime(ir), tokens).Unanchored);
                foreach (var f in item.Forbidden ?? [])
                    if (tokens.Any(t => Morph.CanSurfaceAs(f, t.Text) || string.Equals(f, t.Text, StringComparison.OrdinalIgnoreCase)))
                        forbidden.Add(f);

                if (reparsed?.GetValueOrDefault(item.Id) is { } rp && Linear.TryParse(rp, out var back, out _))
                {
                    var g = TripleSet.Build(ir, tokens);
                    var b = TripleSet.Build(back!, tokens);
                    var m = TripleSet.Matches(g.Triples, b.Triples);
                    foreach (var s in targets) s.RoundTrip.Add(m, b.Triples.Count, g.Triples.Count);
                    rtF1 = g.Triples.Count + b.Triples.Count == 0 ? 1 : 2.0 * m / (g.Triples.Count + b.Triples.Count);
                }
                else if (reparsed is not null)
                {
                    var g = TripleSet.Build(ir, tokens);
                    foreach (var s in targets) s.RoundTrip.Add(0, 0, g.Triples.Count);
                    rtF1 = 0;
                }
            }

            var produced = !string.IsNullOrWhiteSpace(output);
            var auxMismatch = produced ? AuxiliarySignature.Mismatch(Schema.StripRuntime(ir), output!) : null;
            var auxOk = produced && auxMismatch is null;
            var isQuestion = ir.Head is "ynq" or "whq";
            foreach (var s in targets)
            {
                s.Items++;
                s.Produced.Add(produced);
                if (!produced) continue;
                s.HasAddition.Add(additions.Count > 0 || forbidden.Count > 0);
                s.HasForbidden.Add(forbidden.Count > 0);
                s.HasOmission.Add(omissions.Count > 0);
                s.ActPreserved.Add(output!.TrimEnd().EndsWith('?') == isQuestion);
                s.AuxPreserved.Add(auxOk);
                s.AddedTokens += additions.Count;
                s.OutputContentTokens += contentTokens;
            }
            results.Add(new WriteItemResult(item.Id, output, additions, omissions, forbidden, rtF1, auxMismatch));
        }
        return (slices, results);
    }

    static HashSet<string> AllowedLemmas(Node ir) => Coverage.Licensed(ir);
}
