using Language.Core.Data;
using Language.Core.Ir;
using Language.Core.Realize;
using Language.Core.Text;

namespace Language.DataGen;

/// <summary>A training pair for the multitask model. Task is "parse" (text → IR) or "realize" (IR → text).</summary>
public sealed record Pair(string Id, string Task, string Input, string Target);

public sealed record GenStats(int Sampled, int Rejected, int Duplicates, Dictionary<string, int> Acts);

/// <summary>Produces the S1 synthetic dataset: train/dev pairs and an in-distribution held-out test set.</summary>
public static class Generator
{
    public static TextWriter? RejectLog { get; set; }

    public static GenStats Generate(string outDir, int trainCount, int seed)
    {
        Directory.CreateDirectory(outDir);
        var real = Vocab.Verbs.Select(v => v.Lemma)
            .Concat(Vocab.Nouns.Select(n => n.Lemma))
            .Concat(Vocab.Adjectives).Concat(Vocab.MannerAdverbs)
            .Concat(Vocab.Names.Select(n => n.Name));
        var nonsense = new Nonsense(real.Concat(CoreWords.All));
        var poolRng = new Random(seed);
        var trainPool = nonsense.Pool(poolRng, 3000);
        var testPool = nonsense.Pool(poolRng, 600, new HashSet<string>(trainPool));
        File.WriteAllLines(Path.Combine(outDir, "nonsense-train-pool.txt"), trainPool);
        File.WriteAllLines(Path.Combine(outDir, "nonsense-test-pool.txt"), testPool);

        var seen = new HashSet<string>();
        var stats = new GenStats(0, 0, 0, new Dictionary<string, int>());
        var rejected = 0;
        var dups = 0;
        var sampled = 0;

        IEnumerable<(Item Gold, List<Pair> Pairs)> Make(string split, int count, List<string> pool, int s)
        {
            var rng = new Random(s);
            var sampler = new Sampler(rng, pool);
            var made = 0;
            while (made < count)
            {
                sampled++;
                var ir = sampler.Sample();
                var text = Realizer.Realize(ir, RealizerOptions.Sample(rng));
                var problems = Validator.Validate(ir).Concat(Anchoring.Compute(ir, Tokenizer.Tokenize(text)).Unanchored).ToList();
                if (problems.Count > 0)
                {
                    rejected++;
                    RejectLog?.WriteLine($"{text}\n    {Linear.Write(ir)}\n    {string.Join("\n    ", problems)}");
                    continue;
                }
                if (!seen.Add(text)) { dups++; continue; }
                made++;
                stats.Acts[ir.Head] = stats.Acts.GetValueOrDefault(ir.Head) + 1;
                var id = $"s1-{split}-{made:D6}";
                var lin = Linear.Write(ir);
                var hints = sampler.UsedNonsense.Count > 0 && rng.NextDouble() < 0.3
                    ? sampler.UsedNonsense.Select(kv => (kv.Key, kv.Value)).ToList()
                    : null;
                var pairs = new List<Pair>
                {
                    new(id + "-p", "parse", ModelFormat.ParseInput(text, hints), lin),
                    new(id + "-r", "realize", ModelFormat.RealizePrefix + lin, text),
                };
                yield return (new Item { Id = id, Split = split, Category = "synthetic", Text = text, Ir = lin }, pairs);
            }
        }

        // Test first so that its texts are reserved and never reappear in training.
        var test = Make("test", 1000, testPool, seed + 3).ToList();
        var train = Make("train", trainCount, trainPool, seed + 1).ToList();
        var dev = Make("dev", Math.Max(200, trainCount / 20), trainPool, seed + 2).ToList();

        Jsonl.Write(Path.Combine(outDir, "train.jsonl"), train.SelectMany(x => x.Pairs));
        Jsonl.Write(Path.Combine(outDir, "dev.jsonl"), dev.SelectMany(x => x.Pairs));
        Jsonl.Write(Path.Combine(outDir, "test-synthetic.gold.jsonl"), test.Select(x => x.Gold));
        Jsonl.Write(Path.Combine(outDir, "dev.gold.jsonl"), dev.Select(x => x.Gold));
        return stats with { Sampled = sampled, Rejected = rejected, Duplicates = dups };
    }
}
