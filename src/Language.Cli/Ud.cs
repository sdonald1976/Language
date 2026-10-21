using Language.Core.Data;
using Language.Core.Ir;
using Language.DataGen;

namespace Language.Cli;

public static partial class Commands
{
    /// <summary>
    /// Strict: UD treebank → training pairs (S2) for both tasks.
    ///   convert-ud en_ewt-ud-train.conllu --out data/ud/train.jsonl [--gold-out data/ud/train.gold.jsonl]
    /// Lenient (--lenient): automatic parses of test sentences (sent_id = item id) → predictions for baseline B0.
    ///   convert-ud parsed.conllu --lenient --out preds/b0-parse.jsonl
    /// </summary>
    static int ConvertUd(string[] args)
    {
        var inputs = args.TakeWhile(a => !a.StartsWith("--")).ToList();
        if (inputs.Count == 0) throw new ArgumentException("convert-ud <file.conllu>... --out F [--lenient] [--gold-out G]");
        var outPath = Required(args, "--out");
        var lenient = args.Contains("--lenient");
        var conv = new UdConverter(strict: !lenient);
        var pairs = new List<object>();
        var gold = new List<Item>();
        var reasons = new Dictionary<string, int>();
        int total = 0, kept = 0;
        foreach (var path in inputs)
            foreach (var s in UdConverter.ReadConllu(path))
            {
                total++;
                var ir = conv.Convert(s);
                if (lenient)
                {
                    pairs.Add(new { id = s.Id, pred = ir is null ? null : Linear.Write(ir) });
                    if (ir is not null) kept++;
                    continue;
                }
                if (ir is null)
                {
                    var key = conv.LastRejection!.Split(':')[0];
                    reasons[key] = reasons.GetValueOrDefault(key) + 1;
                    continue;
                }
                kept++;
                var lin = Linear.Write(ir);
                var id = "ud-" + s.Id;
                pairs.Add(new { id = id + "-p", task = "parse", input = ModelFormat.ParseInput(s.Text), target = lin });
                pairs.Add(new { id = id + "-r", task = "realize", input = ModelFormat.RealizePrefix + lin, target = s.Text });
                gold.Add(new Item { Id = id, Split = "ud", Category = "ud", Text = s.Text, Ir = lin });
            }
        Jsonl.Write(outPath, pairs);
        if (Opt(args, "--gold-out") is { } g) Jsonl.Write(g, gold);
        Console.WriteLine($"{kept}/{total} sentences converted ({(double)kept / Math.Max(1, total):P1})");
        foreach (var (r, n) in reasons.OrderByDescending(k => k.Value).Take(15)) Console.WriteLine($"  rejected {n,5}: {r}");
        return 0;
    }
}
