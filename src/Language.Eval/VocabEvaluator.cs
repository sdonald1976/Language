using System.Text.Json;
using System.Text.Json.Serialization;
using Language.Core.Ir;
using Language.Core.Text;
using Language.Lexicon;

namespace Language.Eval;

public sealed class Scenario
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("turns")] public List<Turn> Turns { get; set; } = new();
}

public sealed class Turn
{
    [JsonPropertyName("kind")] public string Kind { get; set; } = "read";
    [JsonPropertyName("text")] public string? Text { get; set; }
    [JsonPropertyName("ir")] public string? Ir { get; set; }
    [JsonPropertyName("speaker")] public string? Speaker { get; set; }
    [JsonPropertyName("expect")] public Expect Expect { get; set; } = new();
}

public sealed class Expect
{
    [JsonPropertyName("stored")] public List<StoredExpect>? Stored { get; set; }
    [JsonPropertyName("recognized")] public List<string>? Recognized { get; set; }
    [JsonPropertyName("unknown")] public List<string>? Unknown { get; set; }
    [JsonPropertyName("contains")] public List<string>? Contains { get; set; }
    [JsonPropertyName("absent")] public List<string>? Absent { get; set; }
}

public sealed class StoredExpect
{
    [JsonPropertyName("lemma")] public string Lemma { get; set; } = "";
    [JsonPropertyName("status")] public string Status { get; set; } = "";
}

/// <summary>Vocabulary-learning metrics (DESIGN.md §6.3). Fields omitted from a turn's "expect" are not checked.</summary>
public sealed class VocabScores
{
    public Acc DefinitionStored = new(), NoFalseDefinition = new(), InferredOnly = new(), Recognition = new(), UnknownFlagged = new(),
        GenerationUses = new(), GenerationAvoids = new(), ReadParsed = new(), Scenarios = new();

    public Dictionary<string, object> Summary() => new()
    {
        ["scenarios_fully_correct"] = Scenarios.Value,
        ["definition_extracted_correctly"] = DefinitionStored.Value,
        ["no_false_definition"] = NoFalseDefinition.Value,
        ["inferred_candidate_correct"] = InferredOnly.Value,
        ["learned_word_recognized"] = Recognition.Value,
        ["unknown_word_flagged"] = UnknownFlagged.Value,
        ["generation_uses_learned_word"] = GenerationUses.Value,
        ["generation_avoids_unverified_or_replaced"] = GenerationAvoids.Value,
        ["read_turns_parsed"] = ReadParsed.Value,
    };
}

public static class VocabEvaluator
{
    static readonly HashSet<string> Definitional = [nameof(LexStatus.ExplicitDefinition), nameof(LexStatus.HedgedDefinition), nameof(LexStatus.ReportedDefinition)];

    public static List<Scenario> Load(string path) =>
        File.ReadLines(path).Where(l => !string.IsNullOrWhiteSpace(l))
            .Select(l => JsonSerializer.Deserialize<Scenario>(l)!).ToList();

    /// <param name="read">(turn, this scenario's lexicon) → IR. Pass the gold IR through for the oracle setting, or the engine's reader.</param>
    /// <param name="write">IR (after learned-term substitution) → English.</param>
    public static (VocabScores Scores, List<string> Log) Run(IEnumerable<Scenario> scenarios, Func<Turn, LexiconStore, Node?> read, Func<Node, string> write, string workDir)
    {
        var scores = new VocabScores();
        var log = new List<string>();
        Directory.CreateDirectory(workDir);
        foreach (var sc in scenarios)
        {
            var db = Path.Combine(workDir, $"{sc.Id}.db");
            if (File.Exists(db)) File.Delete(db);
            using var store = new LexiconStore(db);
            var learner = new VocabularyLearner(store);
            var allOk = true;
            foreach (var (turn, i) in sc.Turns.Select((t, i) => (t, i)))
            {
                var where = $"{sc.Id}#{i + 1}";
                if (turn.Kind == "read")
                {
                    var ir = read(turn, store);
                    scores.ReadParsed.Add(ir is not null);
                    if (ir is null) { allOk = false; log.Add($"{where}: reader produced no IR for \"{turn.Text}\""); continue; }
                    var report = learner.Process(ir, turn.Text ?? "", turn.Speaker);
                    var got = report.Stored.Select(s => (s.Lemma.ToLowerInvariant(), s.Status.ToString())).ToHashSet();
                    if (turn.Expect.Stored is { } exp)
                    {
                        var want = exp.Select(s => (s.Lemma.ToLowerInvariant(), s.Status)).ToHashSet();
                        foreach (var w in want)
                        {
                            var ok = got.Contains(w);
                            if (Definitional.Contains(w.Status)) scores.DefinitionStored.Add(ok);
                            else scores.InferredOnly.Add(ok);
                            if (!ok) { allOk = false; log.Add($"{where}: expected {w.Item1}:{w.Item2}, stored [{Fmt(got)}]"); }
                        }
                        var falseDefs = got.Where(g => Definitional.Contains(g.Item2) && !want.Contains(g)).ToList();
                        if (!want.Any(w => Definitional.Contains(w.Status)) || falseDefs.Count > 0)
                            scores.NoFalseDefinition.Add(falseDefs.Count == 0);
                        if (falseDefs.Count > 0) { allOk = false; log.Add($"{where}: unexpected definition(s) [{Fmt(falseDefs)}] from \"{turn.Text}\""); }
                    }
                    if (turn.Expect.Recognized is { } rec)
                        foreach (var r in rec)
                        {
                            var ok = report.Recognized.Any(x => Morph.CanSurfaceAs(x, r) || string.Equals(x, r, StringComparison.OrdinalIgnoreCase)) || report.Recognized.Contains(r, StringComparer.OrdinalIgnoreCase);
                            scores.Recognition.Add(ok);
                            if (!ok) { allOk = false; log.Add($"{where}: '{r}' not recognized (recognized: {string.Join(",", report.Recognized)})"); }
                        }
                    if (turn.Expect.Unknown is { } unk)
                        foreach (var u in unk)
                        {
                            var ok = report.Unknown.Any(x => Morph.CanSurfaceAs(x, u) || string.Equals(x, u, StringComparison.OrdinalIgnoreCase));
                            scores.UnknownFlagged.Add(ok);
                            if (!ok) { allOk = false; log.Add($"{where}: '{u}' not flagged unknown"); }
                        }
                }
                else
                {
                    if (turn.Ir is null) continue;
                    var ir = Linear.Parse(turn.Ir);
                    learner.ApplyLearnedTerms(ir);
                    var text = write(ir);
                    var tokens = Tokenizer.Tokenize(text).Select(t => t.Lower).ToHashSet();
                    foreach (var c in turn.Expect.Contains ?? [])
                    {
                        var ok = tokens.Contains(c.ToLowerInvariant());
                        scores.GenerationUses.Add(ok);
                        if (!ok) { allOk = false; log.Add($"{where}: output \"{text}\" lacks '{c}'"); }
                    }
                    foreach (var a in turn.Expect.Absent ?? [])
                    {
                        var ok = !tokens.Contains(a.ToLowerInvariant());
                        scores.GenerationAvoids.Add(ok);
                        if (!ok) { allOk = false; log.Add($"{where}: output \"{text}\" contains '{a}'"); }
                    }
                }
            }
            scores.Scenarios.Add(allOk);
        }
        return (scores, log);
    }

    static string Fmt(IEnumerable<(string, string)> xs) => string.Join(", ", xs.Select(x => $"{x.Item1}:{x.Item2}"));
}
