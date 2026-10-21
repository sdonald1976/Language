using System.IO.Compression;

namespace Language.Core.Text;

/// <summary>
/// The fixed "already known" English word list (dwyl/english-words, public domain, ~370k forms).
/// A word is unfamiliar when neither it nor any of its candidate lemmas is in this list or the learned lexicon.
/// </summary>
public static class CoreWords
{
    static readonly Lazy<HashSet<string>> Words = new(Load);

    public static HashSet<string> All => Words.Value;

    public static bool IsKnown(string token)
    {
        if (token.Length == 0 || Text.Words.IsPunctuation(token) || token.Any(char.IsDigit)) return true;
        var set = Words.Value;
        if (set.Contains(token.ToLowerInvariant())) return true;
        return Morph.LemmaCandidates(token).Any(set.Contains);
    }

    static HashSet<string> Load()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "core-words.txt.gz");
        var set = new HashSet<string>(StringComparer.Ordinal);
        if (!File.Exists(path)) return set;
        using var gz = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
        using var reader = new StreamReader(gz);
        string? line;
        while ((line = reader.ReadLine()) is not null)
            if (line.Length > 0) set.Add(line);
        // Closed-class words and clitics are always known.
        set.UnionWith(Text.Words.FunctionWords.Select(w => w.ToLowerInvariant()));
        set.UnionWith(["n't", "'s", "'re", "'m", "'ll", "'ve", "'d", "ca", "wo"]);
        return set;
    }
}
