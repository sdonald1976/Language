namespace Language.Core.Text;

/// <summary>Personal pronoun paradigm. Gender: 'm', 'f', 'n' (non-human), or null (human, unspecified).</summary>
public sealed record Pronoun(string Nom, string Acc, string PossDet, string PossPron, string Refl, int Person, bool Plural, char? Gender);

/// <summary>Closed-class English vocabulary. This is grammar, not world knowledge.</summary>
public static class Words
{
    public static readonly Dictionary<string, Pronoun> Pronouns = new()
    {
        ["I"] = new("I", "me", "my", "mine", "myself", 1, false, null),
        ["you"] = new("you", "you", "your", "yours", "yourself", 2, false, null),
        ["he"] = new("he", "him", "his", "his", "himself", 3, false, 'm'),
        ["she"] = new("she", "her", "her", "hers", "herself", 3, false, 'f'),
        ["it"] = new("it", "it", "its", "its", "itself", 3, false, 'n'),
        ["we"] = new("we", "us", "our", "ours", "ourselves", 1, true, null),
        ["they"] = new("they", "them", "their", "theirs", "themselves", 3, true, null),
    };

    /// <summary>Pronoun lemma lookup that accepts any case form ("him" → "he").</summary>
    public static string? PronounLemma(string token)
    {
        var t = token.ToLowerInvariant();
        if (t == "i") return "I";
        foreach (var (lemma, p) in Pronouns)
            if (t == p.Nom.ToLowerInvariant() || t == p.Acc || t == p.PossDet || t == p.PossPron || t == p.Refl)
                return lemma;
        return null;
    }

    public static readonly HashSet<string> Determiners =
    [
        "the", "a", "an", "this", "that", "these", "those", "some", "any", "no", "every", "each", "all", "many",
        "few", "several", "much", "little", "both", "either", "neither", "another", "other", "which", "what", "whose",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twelve", "twenty", "hundred",
        "more", "most", "less", "enough", "such",
    ];

    public static readonly HashSet<string> PluralDeterminers =
        ["these", "those", "many", "few", "several", "both", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twelve", "twenty"];

    public static readonly HashSet<string> Prepositions =
    [
        "in", "on", "at", "to", "from", "with", "without", "by", "for", "of", "about", "under", "over", "near", "behind",
        "beside", "between", "into", "onto", "through", "across", "along", "around", "against", "among", "after",
        "before", "during", "since", "until", "inside", "outside", "above", "below", "toward", "towards", "past", "like",
        "off", "up", "down", "out", "beyond", "next", "front", "despite", "except", "per", "via", "upon", "within",
    ];

    public static readonly HashSet<string> Auxiliaries =
    [
        "be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "having", "do", "does", "did",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must", "wo", "ca", "'s", "'re", "'m", "'ve",
        "'ll", "'d", "n't", "not", "to",
    ];

    public static readonly HashSet<string> Conjunctions =
    [
        "and", "or", "but", "nor", "so", "yet", "if", "unless", "because", "since", "as", "although", "though", "while",
        "when", "whenever", "before", "after", "until", "till", "that", "whether", "than", "once", "even", "then",
    ];

    public static readonly HashSet<string> WhWords = ["who", "whom", "whose", "what", "which", "where", "when", "why", "how"];

    /// <summary>Adverbs that go before the main verb (after the first auxiliary).</summary>
    public static readonly HashSet<string> PreverbalAdverbs =
        ["always", "often", "sometimes", "usually", "rarely", "seldom", "never", "already", "just", "still", "also", "almost", "really", "probably", "even", "only", "ever", "hardly"];

    /// <summary>Function words that a writer may introduce without adding content.</summary>
    public static readonly HashSet<string> FunctionWords = BuildFunctionWords();

    static HashSet<string> BuildFunctionWords()
    {
        var s = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        s.UnionWith(Determiners);
        s.UnionWith(Prepositions);
        s.UnionWith(Auxiliaries);
        s.UnionWith(Conjunctions);
        s.UnionWith(WhWords);
        s.UnionWith(["there", "it", "who", "that", "which", "not", "n't", "to", "of", "'s", "s"]);
        foreach (var p in Pronouns.Values) s.UnionWith([p.Nom, p.Acc, p.PossDet, p.PossPron, p.Refl]);
        return s;
    }

    public static bool IsPunctuation(string token) => token.All(c => char.IsPunctuation(c) || char.IsSymbol(c));
}
