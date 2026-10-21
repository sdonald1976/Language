using System.Text.Json;
using System.Text.Json.Serialization;

namespace Language.Core.Data;

/// <summary>One dataset line. Gold items carry Text and Ir; prediction files carry Id plus Pred.</summary>
public sealed class Item
{
    [JsonPropertyName("id")] public string Id { get; set; } = "";
    [JsonPropertyName("split")] public string? Split { get; set; }
    [JsonPropertyName("category")] public string? Category { get; set; }
    [JsonPropertyName("text")] public string? Text { get; set; }
    [JsonPropertyName("ir")] public string? Ir { get; set; }
    [JsonPropertyName("pred")] public string? Pred { get; set; }
    /// <summary>Gold checks: "knowledge" marks coreference that needs world knowledge (gold: ambiguous).</summary>
    [JsonPropertyName("tags")] public List<string>? Tags { get; set; }
    [JsonPropertyName("notes")] public string? Notes { get; set; }
    /// <summary>Writing tests: words a faithful realization must not contain (tempting additions).</summary>
    [JsonPropertyName("forbidden")] public List<string>? Forbidden { get; set; }
    [JsonPropertyName("verified")] public bool? Verified { get; set; }
}

public static class Jsonl
{
    public static readonly JsonSerializerOptions Options = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public static IEnumerable<T> Read<T>(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"No such file: {path}");
        var lineNo = 0;
        foreach (var line in File.ReadLines(path))
        {
            lineNo++;
            if (string.IsNullOrWhiteSpace(line) || line.TrimStart().StartsWith("//")) continue;
            T? value;
            try { value = JsonSerializer.Deserialize<T>(line, Options); }
            catch (JsonException e) { throw new ArgumentException($"{path}:{lineNo}: {e.Message}"); }
            if (value is not null) yield return value;
        }
    }

    public static void Write<T>(string path, IEnumerable<T> items)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        using var w = new StreamWriter(path);
        foreach (var i in items) w.WriteLine(JsonSerializer.Serialize(i, Options));
    }

    public static string Line<T>(T item) => JsonSerializer.Serialize(item, Options);
}
