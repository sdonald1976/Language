using System.Text.Json;
using Language.Core.Data;
using Language.Core.Ir;
using Language.Core.Realize;
using Language.DataGen;
using Language.Engine;
using Language.Eval;
using Language.Lexicon;

namespace Language.Cli;

public static partial class Commands
{
    static Task<int> RunMore(string command, string[] args) => Task.FromResult(command switch
    {
        "gen" => Gen(args),
        "eval-parse" => EvalParse(args),
        "eval-write" => EvalWrite(args),
        "baseline-write" => BaselineWrite(args),
        "vocab-eval" => VocabEval(args),
        "run" => RunEngine(args),
        "engine-predict" => EnginePredict(args),
        "lexicon" => LexiconCmd(args),
        "generate" => RawGenerate(args),
        "convert-ud" => ConvertUd(args),
        _ => Unknown(command),
    });

    static int Unknown(string command)
    {
        Console.Error.WriteLine($"Unknown command '{command}'");
        return Usage();
    }

    static string? Opt(string[] args, string name, string? fallback = null)
    {
        var i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;
    }

    /// <summary>All values after a flag up to the next flag: --gold a.jsonl b.jsonl</summary>
    static List<string> Many(string[] args, string name)
    {
        var i = Array.IndexOf(args, name);
        var list = new List<string>();
        if (i < 0) return list;
        for (var j = i + 1; j < args.Length && !args[j].StartsWith("--"); j++) list.Add(args[j]);
        return list;
    }

    static string Required(string[] args, string name) => Opt(args, name) ?? throw new ArgumentException($"{name} is required");

    static List<Item> Gold(string[] args)
    {
        var files = Many(args, "--gold");
        if (files.Count == 0) throw new ArgumentException("--gold is required");
        return files.SelectMany(Jsonl.Read<Item>).ToList();
    }

    static Dictionary<string, string?> Predictions(string path) =>
        Jsonl.Read<Item>(path).GroupBy(i => i.Id).ToDictionary(g => g.Key, g => g.First().Pred);

    static readonly JsonSerializerOptions Pretty = new() { WriteIndented = true, NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals };

    static void Report(string? outPath, object report)
    {
        var json = JsonSerializer.Serialize(report, Pretty);
        if (outPath is null) return;
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllText(outPath, json);
        Console.WriteLine($"report written to {outPath}");
    }

    static string F(object? v) => v switch
    {
        double d when double.IsNaN(d) => "  -  ",
        double d => d.ToString("0.000"),
        _ => v?.ToString() ?? "-",
    };

    // ------------------------------------------------------------------ data generation

    static int Gen(string[] args)
    {
        var outDir = Required(args, "--out");
        Directory.CreateDirectory(outDir);
        var n = int.Parse(Opt(args, "--n", "15000")!);
        var seed = int.Parse(Opt(args, "--seed", "1234")!);
        using var log = new StreamWriter(Path.Combine(outDir, "rejected.log"));
        Generator.RejectLog = log;
        var stats = Generator.Generate(outDir, n, seed);
        Console.WriteLine($"sampled {stats.Sampled}, rejected {stats.Rejected} (invalid or unanchored), duplicates {stats.Duplicates}");
        Console.WriteLine("acts: " + string.Join(", ", stats.Acts.OrderBy(k => k.Key).Select(k => $"{k.Key}={k.Value}")));
        return 0;
    }

    // ------------------------------------------------------------------ evaluation

    static int EvalParse(string[] args)
    {
        var gold = Gold(args);
        var preds = Predictions(Required(args, "--pred"));
        var (slices, items) = ParseEvaluator.Evaluate(gold, preds);
        Console.WriteLine($"{"slice",-22} {"n",4} {"wellf",6} {"F1",6} {"exact",6} {"act",6} {"tense",6} {"pol",6} {"invent",7} {"corefR",6} {"corefK",6} {"ambP",6} {"ambR",6}");
        foreach (var (name, s) in slices.OrderBy(k => k.Key == "all" ? "" : k.Key))
            Console.WriteLine($"{name,-22} {s.Items,4} {F(s.WellFormed.Value),6} {F(s.Triples.F1),6} {F(s.Exact.Value),6} {F(s.Act.Value),6} " +
                              $"{F(s.Features["tense"].Value),6} {F(s.Features["pol"].Value),6} {F(s.InventedRate),7} {F(s.CorefResolvable.Value),6} " +
                              $"{F(s.CorefKnowledgeMarkedAmbiguous.Value),6} {F(s.Ambiguity.Precision),6} {F(s.Ambiguity.Recall),6}");
        var errorsPath = Opt(args, "--errors");
        if (errorsPath is not null)
            File.WriteAllLines(errorsPath, items.Where(i => i.Errors.Count > 0 || i.F1 < 1)
                .Select(i => $"{i.Id}\tF1={i.F1:0.00}\t{string.Join(" | ", i.Errors)}"));
        Report(Opt(args, "--out"), new { slices = slices.ToDictionary(k => k.Key, k => k.Value.Summary()), items });
        return 0;
    }

    static int EvalWrite(string[] args)
    {
        var gold = Gold(args);
        var outputs = Predictions(Required(args, "--pred"));
        var reparse = Opt(args, "--reparse") is { } r ? Predictions(r) : null;
        var (slices, items) = WriteEvaluator.Evaluate(gold, outputs, reparse);
        Console.WriteLine($"{"slice",-24} {"n",4} {"added",6} {"forbid",6} {"omit",6} {"type",6} {"roundtripF1",11}");
        foreach (var (name, s) in slices.OrderBy(k => k.Key == "all" ? "" : k.Key))
            Console.WriteLine($"{name,-24} {s.Items,4} {F(s.HasAddition.Value),6} {F(s.HasForbidden.Value),6} {F(s.HasOmission.Value),6} {F(s.ActPreserved.Value),6} {F(s.Summary()["round_trip_triple_f1"]),11}");
        var errorsPath = Opt(args, "--errors");
        if (errorsPath is not null)
            File.WriteAllLines(errorsPath, items.Where(i => i.Additions.Count + i.Omissions.Count + i.Forbidden.Count > 0)
                .Select(i => $"{i.Id}\t{i.Output}\tadded=[{string.Join(",", i.Additions)}] forbidden=[{string.Join(",", i.Forbidden)}] omitted=[{string.Join(",", i.Omissions)}]"));
        Report(Opt(args, "--out"), new { slices = slices.ToDictionary(k => k.Key, k => k.Value.Summary()), items });
        return 0;
    }

    /// <summary>Baseline B0 for writing: the deterministic rule realizer.</summary>
    static int BaselineWrite(string[] args)
    {
        var gold = Gold(args);
        var outPath = Required(args, "--out");
        var lines = new List<Item>();
        foreach (var g in gold.Where(g => g.Ir is not null))
        {
            string? text;
            try { text = Realizer.Realize(Linear.Parse(g.Ir!)); }
            catch (Exception e) when (e is ArgumentException or IrFormatException) { text = null; }
            lines.Add(new Item { Id = g.Id, Pred = text });
        }
        Jsonl.Write(outPath, lines);
        Console.WriteLine($"wrote {lines.Count} realizations to {outPath}");
        return 0;
    }

    static int VocabEval(string[] args)
    {
        var scenarios = VocabEvaluator.Load(Required(args, "--scenarios"));
        var modelDir = Opt(args, "--model");
        var work = Opt(args, "--work", Path.Combine(Path.GetTempPath(), "lang-vocab-eval"))!;
        VocabScores scores;
        List<string> log;
        if (modelDir is null)
        {
            Console.WriteLine("oracle mode: gold IRs stand in for the reader, the rule realizer is the writer");
            (scores, log) = VocabEvaluator.Run(scenarios, (t, _) => t.Ir is null ? null : Linear.Parse(t.Ir), ir => Realizer.Realize(ir), work);
        }
        else
        {
            using var model = new Seq2SeqModel(modelDir);
            var engine = new LanguageEngine(model);
            // The evaluator owns one lexicon per scenario: the reader gets hints from it but does not learn on its own.
            (scores, log) = VocabEvaluator.Run(scenarios,
                (t, store) => t.Text is null ? null : new LanguageEngine(model, store).Read(t.Text, t.Speaker, learn: false).Ir,
                ir => engine.Write(ir, useLearnedTerms: false).Text, work);
        }
        foreach (var (k, v) in scores.Summary()) Console.WriteLine($"{k,-42} {F(v)}");
        foreach (var l in log) Console.WriteLine("  " + l);
        Report(Opt(args, "--out"), new { summary = scores.Summary(), failures = log });
        return 0;
    }

    // ------------------------------------------------------------------ engine

    static int RunEngine(string[] args)
    {
        using var model = new Seq2SeqModel(Required(args, "--model"));
        using var store = Opt(args, "--db") is { } db ? new LexiconStore(db) : null;
        var engine = new LanguageEngine(model, store);
        var rest = args.SkipWhile(a => a != "parse" && a != "realize").ToArray();
        if (rest.Length < 2) throw new ArgumentException("usage: run --model DIR [--db lex.db] parse|realize <text-or-IR>");
        var input = string.Join(' ', rest[1..]);
        if (rest[0] == "parse")
        {
            var r = engine.Read(input);
            Console.WriteLine(r.Ir is null ? $"(no valid IR) raw: {r.RawOutput}" : Linear.Write(r.Ir));
            foreach (var p in r.Problems) Console.WriteLine("  rejected candidate: " + p);
            if (r.Lexicon is { } lx)
            {
                if (lx.Unknown.Count > 0) Console.WriteLine("  unknown words: " + string.Join(", ", lx.Unknown));
                if (lx.Recognized.Count > 0) Console.WriteLine("  learned words recognized: " + string.Join(", ", lx.Recognized));
                foreach (var s in lx.Stored) Console.WriteLine($"  stored: {s.Lemma} ({s.Pos}) = {s.DefinitionText ?? "(no definition)"} [{s.Status}]");
            }
            Console.WriteLine($"  {r.Milliseconds:0} ms");
        }
        else
        {
            var w = engine.Write(Linear.Parse(input));
            Console.WriteLine(w.Text);
            Console.WriteLine($"  source: {w.Source}, learned terms used: {w.LearnedTermsUsed}, {w.Milliseconds:0} ms");
            foreach (var j in w.Rejections) Console.WriteLine("  rejected: " + j);
        }
        return 0;
    }

    /// <summary>Runs the full C# engine (model + checks + fallback) over gold files and writes prediction files.</summary>
    static int EnginePredict(string[] args)
    {
        using var model = new Seq2SeqModel(Required(args, "--model"));
        var engine = new LanguageEngine(model) { Beams = int.Parse(Opt(args, "--beams", "4")!) };
        var task = Required(args, "--task");
        var outPath = Required(args, "--out");
        var lines = new List<object>();
        var ms = new List<double>();
        foreach (var g in Gold(args))
        {
            if (task == "parse" && g.Text is not null)
            {
                var r = engine.Read(g.Text, learn: false);
                ms.Add(r.Milliseconds);
                lines.Add(new { id = g.Id, pred = r.Ir is null ? r.RawOutput : Linear.Write(r.Ir), ms = Math.Round(r.Milliseconds, 1) });
            }
            else if (task == "realize" && g.Ir is not null)
            {
                var w = engine.Write(Linear.Parse(g.Ir), useLearnedTerms: false);
                ms.Add(w.Milliseconds);
                lines.Add(new { id = g.Id, pred = w.Text, source = w.Source, ms = Math.Round(w.Milliseconds, 1) });
            }
        }
        Jsonl.Write(outPath, lines);
        ms.Sort();
        if (ms.Count > 0) Console.WriteLine($"wrote {lines.Count} predictions; latency p50 {ms[ms.Count / 2]:0} ms, p95 {ms[(int)(ms.Count * 0.95)]:0} ms");
        return 0;
    }

    /// <summary>Raw model call with no prefix and no checks, for parity tests against Python.</summary>
    static int RawGenerate(string[] args)
    {
        using var model = new Seq2SeqModel(Required(args, "--model"));
        var beams = int.Parse(Opt(args, "--beams", "4")!);
        var input = string.Join(' ', args.SkipWhile(a => a != "--").Skip(1));
        Console.WriteLine("ids: " + string.Join(",", model.Encode(input)));
        foreach (var (text, score) in model.Generate(input, beams)) Console.WriteLine($"{score:0.000}\t{text}");
        return 0;
    }

    static int LexiconCmd(string[] args)
    {
        if (args.Length < 2) throw new ArgumentException("usage: lexicon <db> list|lookup <word>");
        using var store = new LexiconStore(args[0]);
        switch (args[1])
        {
            case "list":
                foreach (var s in store.AllSenses())
                    Console.WriteLine($"{s.Lemma} ({s.Pos}) [{s.Status}] {s.DefinitionText ?? "-"}  ← \"{s.SourceText}\"{(s.Speaker is null ? "" : " — " + s.Speaker)}");
                return 0;
            case "lookup" when args.Length > 2:
                var e = store.Lookup(args[2]);
                Console.WriteLine(e is null ? "not in lexicon" : $"{e.Lemma} ({e.Pos}) best evidence: {e.BestStatus}, {e.Senses.Count} sense(s), {store.UsageCount(e.Lemma)} usage(s)");
                return 0;
            default:
                throw new ArgumentException("usage: lexicon <db> list|lookup <word>");
        }
    }
}
