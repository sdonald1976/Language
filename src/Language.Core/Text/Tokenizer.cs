using System.Text.RegularExpressions;

namespace Language.Core.Text;

/// <summary>A token with its character span [Start, End) in the original text.</summary>
public sealed record Token(string Text, int Start, int End)
{
    public string Lower => Text.ToLowerInvariant();
}

/// <summary>Penn-style English tokenizer: splits punctuation and clitics ("doesn't" → "does" "n't", "I'm" → "I" "'m").</summary>
public static partial class Tokenizer
{
    [GeneratedRegex(@"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*|[^\sA-Za-z0-9]", RegexOptions.Compiled)]
    private static partial Regex RawToken();

    static readonly string[] Clitics = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"];

    public static List<Token> Tokenize(string text)
    {
        var result = new List<Token>();
        foreach (Match m in RawToken().Matches(text))
        {
            var word = m.Value.Replace('’', '\'');
            var start = m.Index;
            var split = false;
            foreach (var c in Clitics)
            {
                if (!word.EndsWith(c, StringComparison.OrdinalIgnoreCase) || word.Length <= c.Length) continue;
                var cut = word.Length - c.Length;
                var stem = word[..cut];
                if (c == "n't" && stem.Equals("ca", StringComparison.OrdinalIgnoreCase)) stem = "ca";
                result.Add(new Token(stem, start, start + cut));
                result.Add(new Token(word[cut..], start + cut, start + word.Length));
                split = true;
                break;
            }
            if (!split) result.Add(new Token(word, start, start + word.Length));
        }
        return result;
    }
}
