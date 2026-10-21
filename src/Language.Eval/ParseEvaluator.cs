using Language.Core.Data;
using Language.Core.Ir;
using Language.Core.Text;

namespace Language.Eval;

/// <summary>Reading metrics for one slice of the test data (DESIGN.md §6.1).</summary>
public sealed class ParseScores
{
    public int Items;
    public Acc WellFormed = new(), Valid = new(), Exact = new(), Act = new();
    public Prf Triples = new();
    public Dictionary<string, Acc> Features = new() { ["tense"] = new(), ["aspect"] = new(), ["pol"] = new(), ["mod"] = new(), ["voice"] = new() };
    public Acc CorefResolvable = new(), CorefKnowledgeMarkedAmbiguous = new(), CorefAll = new();
    public int InventedItems, ContentItems, ItemsWithInvention;
    public Prf Ambiguity = new();

    public double InventedRate => ContentItems == 0 ? 0 : (double)InventedItems / ContentItems;

    public Dictionary<string, object> Summary() => new()
    {
        ["items"] = Items,
        ["well_formed"] = WellFormed.Value,
        ["valid"] = Valid.Value,
        ["triple_f1"] = Triples.F1,
        ["triple_precision"] = Triples.Precision,
        ["triple_recall"] = Triples.Recall,
        ["exact_match"] = Exact.Value,
        ["act_accuracy"] = Act.Value,
        ["feature_accuracy"] = Features.ToDictionary(k => k.Key, k => (object)k.Value.Value),
        ["invented_content_rate"] = InventedRate,
        ["items_with_invented_content"] = ItemsWithInvention,
        ["coref_accuracy"] = CorefAll.Value,
        ["coref_resolvable_accuracy"] = CorefResolvable.Value,
        ["coref_knowledge_marked_ambiguous"] = CorefKnowledgeMarkedAmbiguous.Value,
        ["ambiguity_precision"] = Ambiguity.Precision,
        ["ambiguity_recall"] = Ambiguity.Recall,
    };
}

public sealed record ItemResult(string Id, string? Category, double F1, bool WellFormed, bool Exact, int Invented, List<string> Errors);

public static class ParseEvaluator
{
    public static (Dictionary<string, ParseScores> Slices, List<ItemResult> Items) Evaluate(IEnumerable<Item> gold, IReadOnlyDictionary<string, string?> predictions)
    {
        var slices = new Dictionary<string, ParseScores>();
        var results = new List<ItemResult>();
        ParseScores Slice(string name) => slices.TryGetValue(name, out var s) ? s : slices[name] = new ParseScores();

        foreach (var item in gold)
        {
            if (item.Text is null || item.Ir is null) continue;
            var targets = new List<ParseScores> { Slice("all"), Slice(item.Category ?? "uncategorized") };
            foreach (var tag in item.Tags ?? []) targets.Add(Slice($"{item.Category}:{tag}"));
            var r = Score(item, predictions.GetValueOrDefault(item.Id), targets);
            results.Add(r);
        }
        return (slices, results);
    }

    static ItemResult Score(Item item, string? pred, List<ParseScores> targets)
    {
        var errors = new List<string>();
        var tokens = Tokenizer.Tokenize(item.Text!);
        var gold = Linear.Parse(item.Ir!);
        var goldSet = TripleSet.Build(gold, tokens);

        Node? p = null;
        var wellFormed = pred is not null && Linear.TryParse(pred, out p, out _);
        if (!wellFormed) errors.Add(pred is null ? "missing prediction" : "malformed IR");
        var valid = p is not null && Validator.Validate(p).Count == 0;
        if (p is not null && !valid) errors.Add("schema: " + string.Join("; ", Validator.Validate(p).Take(3)));

        var predSet = p is not null ? TripleSet.Build(p, tokens) : null;
        var matched = predSet is null ? 0 : TripleSet.Matches(goldSet.Triples, predSet.Triples);
        var exact = p is not null && Canonical(p) == Canonical(gold);
        var anchoring = p is not null ? Anchoring.Compute(Schema.StripRuntime(p), tokens) : null;
        if (anchoring is { Unanchored.Count: > 0 }) errors.Add("invented: " + string.Join(", ", anchoring.Unanchored));

        foreach (var s in targets)
        {
            s.Items++;
            s.WellFormed.Add(wellFormed);
            s.Valid.Add(valid);
            s.Exact.Add(exact);
            s.Act.Add(p is not null && p.Head == gold.Head);
            s.Triples.Add(matched, predSet?.Triples.Count ?? 0, goldSet.Triples.Count);
            if (anchoring is not null)
            {
                s.InventedItems += anchoring.Unanchored.Count;
                s.ContentItems += anchoring.ContentItems;
                if (anchoring.Unanchored.Count > 0) s.ItemsWithInvention++;
            }
            var goldAmbig = gold.Has("ambig");
            var predAmbig = p?.Has("ambig") ?? false;
            s.Ambiguity.Add(goldAmbig && predAmbig ? 1 : 0, predAmbig ? 1 : 0, goldAmbig ? 1 : 0);
        }

        // Clause features and pronoun reference on aligned nodes.
        var predByKey = predSet?.Keys.GroupBy(kv => kv.Value).ToDictionary(g => g.Key, g => g.First().Key) ?? new();
        var goldIdKey = goldSet.Keys.Where(kv => kv.Key.Id is not null).ToDictionary(kv => kv.Key.Id!, kv => kv.Value);
        var predIdKey = predSet?.Keys.Where(kv => kv.Key.Id is not null).GroupBy(kv => kv.Key.Id!).ToDictionary(g => g.Key, g => g.First().Value) ?? new();
        foreach (var (node, key) in goldSet.Keys)
        {
            predByKey.TryGetValue(key, out var pn);
            if (node.IsEvent)
                foreach (var f in new[] { "tense", "aspect", "pol", "mod", "voice" })
                {
                    var ok = pn is not null && pn.IsEvent && node.Atom(f) == pn.Atom(f);
                    foreach (var s in targets) s.Features[f].Add(ok);
                    if (!ok && (node.Atom(f) is not null || pn?.Atom(f) is not null)) errors.Add($"{f} on '{node.Head}': gold {node.Atom(f) ?? "-"}, pred {pn?.Atom(f) ?? "-"}");
                }
            if (node.IsMention && node.Has("ref"))
            {
                var goldRefs = node.Atoms("ref").Select(r => r == "none" ? "none" : goldIdKey.GetValueOrDefault(r, r)).ToHashSet();
                var predRefs = pn?.Atoms("ref").Select(r => r == "none" ? "none" : predIdKey.GetValueOrDefault(r, r)).ToHashSet() ?? new HashSet<string>();
                var ok = goldRefs.SetEquals(predRefs);
                var knowledge = item.Tags?.Contains("knowledge") == true && goldRefs.Count >= 2;
                foreach (var s in targets)
                {
                    s.CorefAll.Add(ok);
                    if (knowledge) s.CorefKnowledgeMarkedAmbiguous.Add(predRefs.Count >= 2);
                    else if (goldRefs.Count == 1) s.CorefResolvable.Add(ok);
                }
                if (!ok) errors.Add($"ref of '{node.Head}': gold {string.Join("+", goldRefs)}, pred {(predRefs.Count == 0 ? "-" : string.Join("+", predRefs))}");
            }
        }

        var f1 = Prf1(matched, predSet?.Triples.Count ?? 0, goldSet.Triples.Count);
        return new ItemResult(item.Id, item.Category, f1, wellFormed, exact, anchoring?.Unanchored.Count ?? 0, errors);
    }

    static double Prf1(int m, int p, int g) => p + g == 0 ? 1 : 2.0 * m / (p + g);

    public static string Canonical(Node n)
    {
        var c = Schema.Canonicalize(Schema.Renumber(Schema.Canonicalize(Schema.StripRuntime(n))));
        return Linear.Write(c);
    }
}
