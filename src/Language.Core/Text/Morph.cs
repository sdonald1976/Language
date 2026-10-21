namespace Language.Core.Text;

/// <summary>English inflectional morphology: generating forms from lemmas, and guessing lemmas from forms.
/// Regular rules cover invented words, so "gorp" → "gorped", "gorping", "gorps".</summary>
public static class Morph
{
    // lemma -> (past, past participle)
    static readonly Dictionary<string, (string Past, string PastPart)> IrregularVerbs = Parse("""
        arise arose arisen|awake awoke awoken|be was been|bear bore borne|beat beat beaten|become became become|
        begin began begun|bend bent bent|bet bet bet|bind bound bound|bite bit bitten|bleed bled bled|blow blew blown|
        break broke broken|breed bred bred|bring brought brought|build built built|burst burst burst|buy bought bought|
        catch caught caught|choose chose chosen|cling clung clung|come came come|cost cost cost|creep crept crept|
        cut cut cut|deal dealt dealt|dig dug dug|do did done|draw drew drawn|drink drank drunk|drive drove driven|
        eat ate eaten|fall fell fallen|feed fed fed|feel felt felt|fight fought fought|find found found|flee fled fled|
        fling flung flung|fly flew flown|forbid forbade forbidden|forget forgot forgotten|forgive forgave forgiven|
        freeze froze frozen|get got gotten|give gave given|go went gone|grind ground ground|grow grew grown|
        hang hung hung|have had had|hear heard heard|hide hid hidden|hit hit hit|hold held held|hurt hurt hurt|
        keep kept kept|kneel knelt knelt|know knew known|lay laid laid|lead led led|leave left left|lend lent lent|
        let let let|lie lay lain|light lit lit|lose lost lost|make made made|mean meant meant|meet met met|
        pay paid paid|put put put|quit quit quit|read read read|ride rode ridden|ring rang rung|rise rose risen|
        run ran run|say said said|see saw seen|seek sought sought|sell sold sold|send sent sent|set set set|
        shake shook shaken|shine shone shone|shoot shot shot|show showed shown|shrink shrank shrunk|shut shut shut|
        sing sang sung|sink sank sunk|sit sat sat|sleep slept slept|slide slid slid|speak spoke spoken|
        spend spent spent|spin spun spun|split split split|spread spread spread|spring sprang sprung|stand stood stood|
        steal stole stolen|stick stuck stuck|sting stung stung|stink stank stunk|strike struck struck|
        swear swore sworn|sweep swept swept|swim swam swum|swing swung swung|take took taken|teach taught taught|
        tear tore torn|tell told told|think thought thought|throw threw thrown|understand understood understood|
        wake woke woken|wear wore worn|weep wept wept|win won won|wind wound wound|write wrote written
        """);

    static readonly Dictionary<string, string> IrregularPlurals = new()
    {
        ["man"] = "men", ["woman"] = "women", ["child"] = "children", ["person"] = "people", ["foot"] = "feet",
        ["tooth"] = "teeth", ["mouse"] = "mice", ["goose"] = "geese", ["sheep"] = "sheep", ["fish"] = "fish",
        ["deer"] = "deer", ["ox"] = "oxen", ["knife"] = "knives", ["wife"] = "wives", ["life"] = "lives",
        ["leaf"] = "leaves", ["wolf"] = "wolves", ["half"] = "halves", ["shelf"] = "shelves", ["loaf"] = "loaves",
        ["thief"] = "thieves", ["potato"] = "potatoes", ["tomato"] = "tomatoes", ["hero"] = "heroes",
        ["calf"] = "calves", ["policeman"] = "policemen", ["fireman"] = "firemen", ["series"] = "series",
    };

    static readonly HashSet<string> NoDoubling = ["visit", "open", "listen", "happen", "offer", "answer", "enter", "order",
        "travel", "cancel", "label", "model", "signal", "wonder", "gather", "cover", "deliver", "remember", "consider",
        "fix", "mix", "box", "show", "snow", "allow", "follow", "borrow", "play", "stay", "enjoy", "destroy",
        "edit", "limit", "benefit", "credit", "exit", "orbit", "develop", "gallop", "budget", "target", "market",
        "water", "paint", "rain", "cook", "look", "wait", "need", "clean", "dream", "laugh", "help", "kick", "walk",
        "cheat", "eat", "heat", "shout", "doubt", "count", "rest", "test", "lift", "print", "push", "pull",
        "suffer", "differ", "bother", "matter", "murmur", "visit"];

    static readonly HashSet<string> ForceDoubling = ["admit", "commit", "permit", "omit", "submit", "prefer", "refer",
        "occur", "regret", "forget", "begin", "control", "patrol", "compel", "propel", "expel", "equip", "upset", "transfer"];

    static Dictionary<string, (string, string)> Parse(string table)
    {
        var d = new Dictionary<string, (string, string)>();
        foreach (var entry in table.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var w = entry.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            d[w[0]] = (w[1], w[2]);
        }
        return d;
    }

    static bool IsVowel(char c) => "aeiou".Contains(c);

    static int SyllableGuess(string w)
    {
        var n = 0;
        var prev = false;
        foreach (var c in w)
        {
            var v = IsVowel(c) || c == 'y';
            if (v && !prev) n++;
            prev = v;
        }
        if (w.EndsWith('e') && !w.EndsWith("le") && n > 1) n--;
        return Math.Max(n, 1);
    }

    /// <summary>Should the final consonant double before -ed/-ing? (stop→stopped, hug→hugging)</summary>
    public static bool DoublesFinal(string verb)
    {
        if (ForceDoubling.Contains(verb)) return true;
        if (NoDoubling.Contains(verb) || verb.Length < 3) return false;
        var a = verb[^3]; var b = verb[^2]; var c = verb[^1];
        return !IsVowel(a) && IsVowel(b) && !IsVowel(c) && !"wxy".Contains(c)
               && !(a == 'q' || verb.Length > 3 && IsVowel(verb[^4]))
               && SyllableGuess(verb) == 1;
    }

    static readonly string[] VerbPrefixes = ["over", "under", "re", "mis", "out", "with", "fore", "up", "un"];

    /// <summary>Irregular forms, including prefixed irregulars ("oversleep" → "overslept", "rewrite" → "rewrote").</summary>
    static bool TryIrregular(string verb, out (string Past, string PastPart) forms)
    {
        if (IrregularVerbs.TryGetValue(verb, out forms)) return true;
        foreach (var p in VerbPrefixes)
            if (verb.Length > p.Length + 2 && verb.StartsWith(p) && IrregularVerbs.TryGetValue(verb[p.Length..], out var f))
            {
                forms = (p + f.Past, p + f.PastPart);
                return true;
            }
        return false;
    }

    public static string Past(string verb) => TryIrregular(verb, out var f) ? f.Past : RegularEd(verb);

    public static string PastParticiple(string verb) => TryIrregular(verb, out var f) ? f.PastPart : RegularEd(verb);

    static string RegularEd(string v)
    {
        if (v.EndsWith('e')) return v + "d";
        if (v.EndsWith('y') && v.Length > 1 && !IsVowel(v[^2])) return v[..^1] + "ied";
        if (v.EndsWith('c')) return v + "ked";
        if (DoublesFinal(v)) return v + v[^1] + "ed";
        return v + "ed";
    }

    public static string PresentParticiple(string v)
    {
        if (v is "be" or "see" or "flee" or "agree" or "free") return v + "ing";
        if (v.EndsWith("ie")) return v[..^2] + "ying";
        if (v.EndsWith('e') && !v.EndsWith("ee") && !v.EndsWith("ye") && !v.EndsWith("oe") && v.Length > 2) return v[..^1] + "ing";
        if (v.EndsWith('c')) return v + "king";
        if (DoublesFinal(v)) return v + v[^1] + "ing";
        return v + "ing";
    }

    public static string ThirdSingular(string v)
    {
        if (v == "be") return "is";
        if (v == "have") return "has";
        if (v is "do" or "go") return v + "es";
        return Sibilant(v);
    }

    static string Sibilant(string w)
    {
        if (w.EndsWith('s') || w.EndsWith('x') || w.EndsWith('z') || w.EndsWith("ch") || w.EndsWith("sh")) return w + "es";
        if (w.EndsWith('y') && w.Length > 1 && !IsVowel(w[^2])) return w[..^1] + "ies";
        return w + "s";
    }

    public static string Plural(string noun)
    {
        var lower = noun.ToLowerInvariant();
        if (IrregularPlurals.TryGetValue(lower, out var p)) return MatchCase(noun, p);
        return Sibilant(noun);
    }

    static string MatchCase(string model, string w) =>
        model.Length > 0 && char.IsUpper(model[0]) ? char.ToUpperInvariant(w[0]) + w[1..] : w;

    /// <summary>Inflected surface forms of "be" for the given tense/person/number.</summary>
    public static string Be(bool past, int person, bool plural) => (past, person, plural) switch
    {
        (false, 1, false) => "am",
        (false, _, true) or (false, 2, _) => "are",
        (false, _, _) => "is",
        (true, _, true) or (true, 2, _) => "were",
        (true, _, _) => "was",
    };

    /// <summary>All forms a lemma can surface as. Used to anchor IR heads to input tokens.</summary>
    public static HashSet<string> Forms(string lemma)
    {
        var l = lemma.ToLowerInvariant();
        var set = new HashSet<string> { l };
        if (l.Length == 0) return set;
        if (l == "be") set.UnionWith(["am", "is", "are", "was", "were", "been", "being", "'m", "'re", "'s"]);
        if (l == "have") set.UnionWith(["'ve", "'s", "'d"]);
        if (l == "will") set.UnionWith(["'ll", "wo"]);
        if (l == "would") set.Add("'d");
        if (l == "can") set.Add("ca");
        if (l == "not") set.Add("n't");
        if (l is "a" or "an") set.UnionWith(["a", "an"]);
        if (Words.PronounLemma(l) is { } pl && pl.Equals(lemma, StringComparison.OrdinalIgnoreCase) && Words.Pronouns.TryGetValue(pl, out var pr))
            set.UnionWith([pr.Nom.ToLowerInvariant(), pr.Acc, pr.PossDet, pr.PossPron, pr.Refl]);
        if (!l.All(c => char.IsLetter(c) || c == '-' || c == '\'')) return set;
        set.Add(Past(l));
        set.Add(PastParticiple(l));
        set.Add(PresentParticiple(l));
        set.Add(ThirdSingular(l).ToLowerInvariant());
        set.Add(Plural(l).ToLowerInvariant());
        set.Add(l + "'s");
        if (l.EndsWith('l') && l.Length > 2 && IsVowel(l[^2]) && !IsVowel(l[^3]))
            set.UnionWith([l + "led", l + "ling"]); // British spelling: cancelled, travelling
        return set;
    }

    public static bool CanSurfaceAs(string lemma, string token) =>
        string.Equals(lemma, token, StringComparison.OrdinalIgnoreCase) || Forms(lemma).Contains(token.ToLowerInvariant());

    /// <summary>Candidate lemmas for a surface token (over-generates; used for lexicon lookup).</summary>
    public static HashSet<string> LemmaCandidates(string token)
    {
        var t = token.ToLowerInvariant();
        var set = new HashSet<string> { t };
        foreach (var (lemma, f) in IrregularVerbs)
        {
            if (f.Past == t || f.PastPart == t) set.Add(lemma);
            foreach (var p in VerbPrefixes)
                if (t.StartsWith(p) && (p + f.Past == t || p + f.PastPart == t)) set.Add(p + lemma);
        }
        foreach (var (sg, pl) in IrregularPlurals)
            if (pl == t) set.Add(sg);
        if (t is "am" or "is" or "are" or "was" or "were" or "been" or "being") set.Add("be");
        if (t == "has") set.Add("have");
        if (t == "does") set.Add("do");
        if (t == "goes") set.Add("go");
        if (t.EndsWith("ies") && t.Length > 4) set.Add(t[..^3] + "y");
        if (t.EndsWith("es") && t.Length > 3) set.Add(t[..^2]);
        if (t.EndsWith('s') && !t.EndsWith("ss") && t.Length > 2) set.Add(t[..^1]);
        if (t.EndsWith("ied") && t.Length > 4) set.Add(t[..^3] + "y");
        if (t.EndsWith("ed") && t.Length > 3)
        {
            var stem = t[..^2];
            set.Add(stem);
            set.Add(stem + "e");
            if (stem.Length > 2 && stem[^1] == stem[^2]) set.Add(stem[..^1]);
            if (stem.EndsWith("ck")) set.Add(stem[..^1]);
        }
        if (t.EndsWith("d") && t.Length > 2 && t[^2] == 'e') set.Add(t[..^1]);
        if (t.EndsWith("ing") && t.Length > 4)
        {
            var stem = t[..^3];
            set.Add(stem);
            set.Add(stem + "e");
            if (stem.EndsWith('y')) set.Add(stem[..^1] + "ie");
            if (stem.Length > 2 && stem[^1] == stem[^2]) set.Add(stem[..^1]);
            if (stem.EndsWith("ck")) set.Add(stem[..^1]);
        }
        if (t.EndsWith("'s")) set.Add(t[..^2]);
        if (t.EndsWith("s'")) set.Add(t[..^1]);
        foreach (var (lemma, pr) in Words.Pronouns)
            if (t == pr.Nom || t == pr.Acc || t == pr.PossDet || t == pr.PossPron || t == pr.Refl) set.Add(lemma);
        return set;
    }

    /// <summary>"a" or "an" before the given word (spelling-based heuristic with common exceptions).</summary>
    public static string IndefiniteArticle(string nextWord)
    {
        var w = nextWord.ToLowerInvariant();
        if (w.Length == 0) return "a";
        string[] anExceptions = ["hour", "honest", "honor", "heir", "herb"];
        string[] aExceptions = ["uni", "use", "usu", "uti", "one", "once", "euro", "ewe", "uk", "ufo", "uter"];
        if (anExceptions.Any(w.StartsWith)) return "an";
        if (aExceptions.Any(w.StartsWith)) return "a";
        return "aeiou".Contains(w[0]) ? "an" : "a";
    }
}
