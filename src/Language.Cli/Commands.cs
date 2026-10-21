using Language.Core.Ir;
using Language.Core.Realize;
using Language.Core.Text;

namespace Language.Cli;

public static partial class Commands
{
    public static async Task<int> Run(string[] args)
    {
        if (args.Length == 0) return Usage();
        var rest = args[1..];
        try
        {
            return args[0] switch
            {
                "realize" => Realize(rest),
                "check" => Check(rest),
                _ => await RunMore(args[0], rest),
            };
        }
        catch (Exception e) when (e is IrFormatException or ArgumentException or FileNotFoundException)
        {
            Console.Error.WriteLine("error: " + e.Message);
            return 1;
        }
    }

    static int Usage()
    {
        Console.WriteLine("""
            Usage: lang <command> [options]

              realize [IR]                 Rule-based IR → English (reads IR lines from stdin if no argument)
              check <items.jsonl> [--show] Validate gold items: IR well-formed, anchored to text (--show prints rule realizations)
              gen --out DIR [--n N] [--seed S]
                                           Generate the synthetic training/dev/test sets (S1)
              eval-parse --gold G --pred P [--out report.json]
                                           Score parser predictions against gold
              eval-write --gold G --pred P [--reparse R] [--out report.json]
                                           Score realizer outputs for additions/omissions (and round-trip if R given)
              baseline-parse --gold G --out P
                                           Run the rule-based reader (B0) on gold items
              baseline-write --gold G --out P
                                           Run the rule-based realizer (B0) on gold IRs
              lexicon <db> learn|lookup|list ...
                                           Inspect or drive the persistent lexicon
              run --model DIR [--db lex.db] parse|realize <text-or-IR>
                                           Run the neural engine (ONNX) end to end
              vocab-eval --model DIR --scenarios F [--out report.json]
                                           Run vocabulary-learning scenarios (T9)
              convert-ud <file.conllu> --out F
                                           Convert UD English EWT to IR pairs (S2)
            """);
        return 2;
    }

    static int Realize(string[] args)
    {
        var lines = args.Length > 0 ? [string.Join(' ', args)] : ReadStdin();
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var node = Linear.Parse(line);
            var errors = Validator.Validate(node);
            foreach (var e in errors) Console.Error.WriteLine("invalid: " + e);
            Console.WriteLine(Realizer.Realize(node));
        }
        return 0;
    }

    static IEnumerable<string> ReadStdin()
    {
        string? l;
        while ((l = Console.ReadLine()) is not null) yield return l;
    }

    /// <summary>Validates a JSONL file of gold items ({"id","text","ir",...}).</summary>
    static int Check(string[] args)
    {
        if (args.Length < 1) return Usage();
        var bad = 0;
        var n = 0;
        var show = args.Contains("--show");
        foreach (var item in Data.ReadItems(args[0]))
        {
            if (item.Ir is null) continue;
            n++;
            var problems = new List<string>();
            if (!Linear.TryParse(item.Ir, out var node, out var err)) problems.Add("parse: " + err);
            else
            {
                problems.AddRange(Validator.Validate(node!).Select(e => "invalid: " + e));
                if (item.Text is not null)
                {
                    var a = Anchoring.Compute(node!, Tokenizer.Tokenize(item.Text));
                    problems.AddRange(a.Unanchored.Select(u => "unanchored: " + u));
                }
            }
            if (problems.Count == 0)
            {
                if (show) Console.WriteLine($"{item.Id}: {item.Text}\n    ok → {Realizer.Realize(node!)}");
                continue;
            }
            bad++;
            Console.WriteLine($"{item.Id}: {item.Text}");
            foreach (var p in problems) Console.WriteLine("    " + p);
            if (node is not null)
            {
                try { Console.WriteLine("    rule realizer: " + Realizer.Realize(node)); }
                catch (Exception e) { Console.WriteLine("    rule realizer failed: " + e.Message); }
            }
        }
        Console.WriteLine($"{n - bad}/{n} items OK");
        return bad == 0 ? 0 : 1;
    }
}
