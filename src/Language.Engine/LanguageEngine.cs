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
        var input = ModelFormat.ParseInput(text, hints);
        var tokens = Tokenizer.Tokenize(text);
        var problems = new List<string>();
        // Greedy first (fast); widen to the beam only if the greedy reading fails a check.
        var (chosen, raw) = Choose(model.Generate(input, 1), tokens, problems, out var fallback);
        if (chosen is null && Beams > 1)
            (chosen, raw) = Choose(model.Generate(input, Beams), tokens, problems, out fallback);
        // A reading that omits content words is used only when nothing better exists, and is reported.
        if (chosen is null && fallback is not null)
        {
            (chosen, raw) = fallback.Value;
            problems.Add("accepted a reading that omits: " + string.Join(", ", Coverage.Uncovered(chosen!, tokens)));
        }
        LearnReport? report = null;
        if (chosen is not null && _learner is not null && learn) report = _learner.Process(chosen, text, speaker);
        return new ReadResult(text, chosen, raw, problems, report, sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>First hypothesis that is well formed, invents nothing and omits nothing. The best one that only
    /// omits something is returned separately as a last resort.</summary>
    static (Node? Node, string? Raw) Choose(List<(string Text, double Score)> outputs, List<Token> tokens, List<string> problems,
        out (Node? Node, string? Raw)? omitting)
    {
        omitting = null;
        foreach (var (candidate, _) in outputs)
        {
            if (!Linear.TryParse(candidate, out var node, out var err)) { problems.Add($"malformed: {err}"); continue; }
            var invalid = Validator.Validate(node!);
            if (invalid.Count > 0) { problems.Add("invalid: " + invalid[0]); continue; }
            var anchoring = Anchoring.Compute(node!, tokens);
            if (anchoring.Unanchored.Count > 0) { problems.Add("invented: " + string.Join(", ", anchoring.Unanchored)); continue; }
            var missing = Coverage.Uncovered(node!, tokens).ToList();
            if (missing.Count > 0)
            {
                problems.Add("omits: " + string.Join(", ", missing));
                omitting ??= (node, candidate);
                continue;
            }
            return (node, candidate);
        }
        return (null, outputs.FirstOrDefault().Text);
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
        var added = Coverage.Uncovered(ir, tokens).ToList();
        if (added.Count > 0) return "adds " + string.Join(", ", added);
        if (AuxiliarySignature.Mismatch(ir, text) is { } aux) return aux;
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
