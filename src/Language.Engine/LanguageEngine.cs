using System.Diagnostics;
using Language.Core.Ir;
using Language.Core.Realize;
using Language.Core.Text;
using Language.Core.Data;
using Language.Lexicon;

namespace Language.Engine;

public sealed record ReadResult(string Text, Node? Ir, string? RawOutput, List<string> Problems, LearnReport? Lexicon, double Milliseconds);

public sealed record WriteResult(string Text, string Source, int CandidateRank, List<string> Rejections, int LearnedTermsUsed, double Milliseconds);

/// <summary>
/// The language engine: English → IR (reading) and IR → English (writing).
/// The neural model proposes; deterministic C# checks dispose:
///  reading rejects malformed IR and content not anchored in the input,
///  writing re-parses every candidate and rejects any that adds or drops IR content,
///  falling back to the rule-based realizer, which is faithful by construction.
/// </summary>
public sealed class LanguageEngine(Seq2SeqModel model, LexiconStore? lexicon = null)
{
    readonly VocabularyLearner? _learner = lexicon is null ? null : new VocabularyLearner(lexicon);

    public int Beams { get; init; } = 4;

    public ReadResult Read(string text, string? speaker = null, bool learn = true)
    {
        var sw = Stopwatch.StartNew();
        var hints = _learner?.Hints(text);
        var outputs = model.Generate(ModelFormat.ParseInput(text, hints), Beams);
        var tokens = Tokenizer.Tokenize(text);
        var problems = new List<string>();
        Node? chosen = null;
        string? raw = outputs.FirstOrDefault().Text;
        // Take the best-scoring hypothesis that is well formed and invents nothing.
        foreach (var (candidate, _) in outputs)
        {
            if (!Linear.TryParse(candidate, out var node, out var err)) { problems.Add($"malformed: {err}"); continue; }
            var invalid = Validator.Validate(node!);
            if (invalid.Count > 0) { problems.Add("invalid: " + invalid[0]); continue; }
            var anchoring = Anchoring.Compute(node!, tokens);
            if (anchoring.Unanchored.Count > 0) { problems.Add("invented: " + string.Join(", ", anchoring.Unanchored)); continue; }
            chosen = node;
            raw = candidate;
            break;
        }
        LearnReport? report = null;
        if (chosen is not null && _learner is not null && learn) report = _learner.Process(chosen, text, speaker);
        return new ReadResult(text, chosen, raw, problems, report, sw.Elapsed.TotalMilliseconds);
    }

    public WriteResult Write(Node ir, bool useLearnedTerms = true)
    {
        var sw = Stopwatch.StartNew();
        ir = ir.Clone();
        var learned = useLearnedTerms && _learner is not null ? _learner.ApplyLearnedTerms(ir) : 0;
        var target = Schema.Canonicalize(Schema.StripRuntime(ir));
        var rejections = new List<string>();
        var candidates = model.Generate(ModelFormat.RealizeInput(target), Beams);
        for (var i = 0; i < candidates.Count; i++)
        {
            var text = candidates[i].Text.Trim();
            var why = Verify(target, text);
            if (why is null) return new WriteResult(text, "model", i, rejections, learned, sw.Elapsed.TotalMilliseconds);
            rejections.Add($"\"{text}\": {why}");
        }
        var fallback = Realizer.Realize(target);
        return new WriteResult(fallback, "rule-fallback", -1, rejections, learned, sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>Returns null when <paramref name="text"/> expresses exactly the IR, or the reason it does not.</summary>
    public string? Verify(Node ir, string text)
    {
        var tokens = Tokenizer.Tokenize(text);
        var omitted = Anchoring.Compute(ir, tokens).Unanchored;
        if (omitted.Count > 0) return "omits " + string.Join(", ", omitted);
        var back = model.Generate(ModelFormat.ParseInput(text), 1).FirstOrDefault().Text;
        if (back is null || !Linear.TryParse(back, out var parsed, out _)) return "re-parse failed";
        var a = Normalize(ir);
        var b = Normalize(parsed!);
        return a == b ? null : $"re-parse differs: {b}";
    }

    /// <summary>Comparison form: canonical, renumbered, without runtime marks or surface-only flags.</summary>
    static string Normalize(Node n)
    {
        var c = Schema.StripRuntime(n);
        c.Props.RemoveAll(p => p.Role == "ambig");
        return Linear.Write(Schema.Canonicalize(Schema.Renumber(Schema.Canonicalize(c))));
    }
}
