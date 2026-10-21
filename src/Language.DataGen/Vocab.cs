namespace Language.DataGen;

/// <summary>Noun classes used only to keep pronoun agreement linguistically correct (he/she/it/they).
/// Unknown = an invented noun, whose animacy the language alone does not reveal.</summary>
public enum NounClass { Human, Animal, Thing, Place, Mass, Unknown }

public sealed record NounEntry(string Lemma, NounClass Class, char? Gender = null);
public sealed record NameEntry(string Name, char? Gender);

/// <summary>Verb frames: I intransitive, T transitive, D ditransitive, C finite complement, X infinitival complement.</summary>
public sealed record VerbEntry(string Lemma, string Frames, bool AnimateSubject);

/// <summary>Real-word vocabulary for the synthetic generator. Plausibility is deliberately not enforced:
/// "The table admired the idea" is a fine training sentence for a language-only engine.</summary>
public static class Vocab
{
    public static readonly VerbEntry[] Verbs = ParseVerbs("""
        sleep I a|run I a|arrive I a|laugh I a|cry I a|fall I -|swim I a|wait IX a|smile I a|work I a|sing IT a|
        dance I a|rest I a|shout I a|jump I a|walk I a|listen I a|leave IT a|disappear I -|shine I -|break IT -|
        open IT -|close IT -|move IT -|stop IT -|start TX -|grow IT -|melt I -|float I -|bark I a|roll IT -|
        eat IT a|see TC a|find TC a|buy TD a|sell TD a|build T a|carry T a|clean T a|cook TD a|paint T a|
        read TD a|write TD a|watch T a|hear TC a|help TX a|like TX a|love TX a|hate TX a|need TX a|want TX a|
        hug T a|push T a|pull T a|kick T a|catch T a|throw TD a|lose T a|keep T a|hold T a|fix T a|wash T a|
        visit T a|call T a|follow T a|meet T a|teach TD a|give D a|send D a|show DC a|tell DC a|bring D a|
        offer D a|lend D a|hand D a|pass D a|say C a|think C a|know TC a|believe C a|hope CX a|notice TC a|
        remember TCX a|forget TCX a|promise CX a|decide CX a|try X a|refuse X a|learn TX a|plan X a|
        drop T a|grab T a|hide T a|feed T a|fill T -|cover T -|touch T a|lift T a|draw T a|drink T a|
        explain T a|answer T a|ask T a|thank T a|invite T a|chase T a|bite T a|steal T a|drive T a|ride T a|
        """);

    public static readonly NounEntry[] Nouns =
    [
        .. Nouns_("man woman boy girl king queen mother father brother sister uncle aunt", NounClass.Human, true),
        .. Nouns_("child teacher doctor farmer student baker nurse friend neighbor pilot driver artist singer " +
                  "student guest stranger cook lawyer soldier writer worker child customer visitor", NounClass.Human),
        .. Nouns_("dog cat horse bird fish mouse cow rabbit fox wolf sheep duck frog bee owl goat", NounClass.Animal),
        .. Nouns_("book table box car door window cup letter key ball chair bike house phone bag apple cake hat " +
                  "lamp bottle coat ticket picture shoe clock basket boat bed candle map pencil plate rope stone " +
                  "drum flag card ring bridge tower wall fence", NounClass.Thing),
        .. Nouns_("park kitchen garden city school river station store office beach forest room village hill " +
                  "street library market island field", NounClass.Place),
        .. Nouns_("water bread milk rice music sand money furniture snow tea coffee news advice", NounClass.Mass),
    ];

    static IEnumerable<NounEntry> Nouns_(string list, NounClass c, bool gendered = false)
    {
        var words = list.Split(' ', StringSplitOptions.RemoveEmptyEntries).Distinct();
        foreach (var w in words)
        {
            char? g = gendered
                ? w is "man" or "boy" or "king" or "father" or "brother" or "uncle" ? 'm' : 'f'
                : null;
            yield return new NounEntry(w, c, g);
        }
    }

    /// <summary>Names carry lexical gender where it is conventional; unisex names have none.</summary>
    public static readonly NameEntry[] Names =
    [
        .. "Tom Bill John David Omar Leo Kenji Carlos Ivan Ahmed Peter Luke".Split(' ').Select(n => new NameEntry(n, 'm')),
        .. "Maria Anna Lena Priya Sarah Emma Fatima Yuki Olga Grace Nora Julia".Split(' ').Select(n => new NameEntry(n, 'f')),
        .. "Alex Sam Jordan Taylor Robin Kim Charlie Morgan".Split(' ').Select(n => new NameEntry(n, null)),
    ];

    public static readonly string[] Adjectives =
        ("big small old new red blue green happy sad quiet loud tall short heavy empty full warm cold bright dark " +
         "clean dirty tired angry quick slow strange broken young soft hard wet dry hungry busy brave lazy").Split(' ');

    public static readonly string[] Intensifiers = ["very", "quite", "really", "too", "extremely"];
    public static readonly string[] MannerAdverbs = ["quickly", "slowly", "quietly", "carefully", "loudly", "happily", "badly", "well", "suddenly", "gently"];
    public static readonly string[] TimeAdverbs = ["yesterday", "today", "tomorrow", "now", "later", "again", "early", "late"];
    public static readonly string[] FrequencyAdverbs = ["always", "often", "sometimes", "usually", "rarely"];

    public static readonly string[] PlacePreps = ["in", "on", "under", "near", "behind", "at", "beside", "inside"];
    public static readonly string[] OtherPreps = ["with", "for", "from", "to", "about", "without", "after"];

    public static readonly string[] Determiners = ["the", "the", "the", "the", "a", "a", "this", "that", "some", "every", "no", "each"];

    static VerbEntry[] ParseVerbs(string table) =>
        table.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(e => e.Split(' '))
            .Select(p => new VerbEntry(p[0], p[1], p[2] == "a"))
            .ToArray();
}
