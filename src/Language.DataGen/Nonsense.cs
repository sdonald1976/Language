namespace Language.DataGen;

/// <summary>Generates pronounceable invented words ("blick", "zindle", "florp") for Jabberwocky augmentation.
/// Train and test draw from disjoint pools, so test-time invented words are never seen in training.</summary>
public sealed class Nonsense
{
    static readonly string[] Onsets = ["b", "bl", "br", "d", "dr", "f", "fl", "fr", "g", "gl", "gr", "k", "kl", "kr", "m", "n", "p", "pl", "pr",
        "s", "sk", "sl", "sn", "sp", "st", "t", "tr", "v", "w", "z", "zw", "j", "ch", "sh", "th", "h", "l", "r", "qu", "squ", "spl"];
    static readonly string[] Vowels = ["a", "e", "i", "o", "u", "oo", "ee", "ai", "ou", "oa"];
    static readonly string[] Codas = ["b", "d", "g", "k", "m", "n", "p", "t", "x", "nd", "nk", "mp", "rp", "rk", "sk", "st", "ng", "rg", "ft", "lt", "ck", "zz", "sh", "rb", "lm"];
    static readonly string[] Endings = ["", "", "", "", "le", "er", "et", "in", "ix", "o", "y", "tron", "ble"];

    readonly HashSet<string> _forbidden;

    public Nonsense(IEnumerable<string> realWords) =>
        _forbidden = new HashSet<string>(realWords.Select(w => w.ToLowerInvariant()));

    public string Make(Random rng)
    {
        while (true)
        {
            var syllables = rng.NextDouble() < 0.75 ? 1 : 2;
            var w = "";
            for (var i = 0; i < syllables; i++)
                w += Pick(rng, Onsets) + Pick(rng, Vowels) + (i == syllables - 1 || rng.NextDouble() < 0.5 ? Pick(rng, Codas) : "");
            w += Pick(rng, Endings);
            if (w.Length < 3 || w.Length > 11 || _forbidden.Contains(w) || CommonWords.Contains(w)) continue;
            if (w.EndsWith('s') || w.EndsWith("ed") || w.EndsWith("ing") || w.EndsWith("ly")) continue;
            return w;
        }
    }

    /// <summary>Builds a pool of distinct invented words; pools for different splits are kept disjoint.</summary>
    public List<string> Pool(Random rng, int size, ISet<string>? exclude = null)
    {
        var set = new HashSet<string>();
        while (set.Count < size)
        {
            var w = Make(rng);
            if (exclude is null || !exclude.Contains(w)) set.Add(w);
        }
        return set.ToList();
    }

    static T Pick<T>(Random rng, T[] a) => a[rng.Next(a.Length)];

    // Short real words the syllable grammar can accidentally produce.
    static readonly HashSet<string> CommonWords =
    [
        "bad", "bed", "big", "bit", "bog", "bud", "bug", "bun", "but", "dig", "dog", "dot", "fan", "fat", "fit", "fog", "fun",
        "gap", "gut", "kid", "kin", "kit", "man", "map", "men", "met", "mop", "mud", "nap", "net", "not", "nut", "pan", "pen",
        "pet", "pig", "pin", "pit", "pot", "put", "sad", "sat", "set", "sit", "son", "sun", "tan", "ten", "tin", "top", "tub",
        "van", "vet", "wet", "win", "wit", "zip", "stop", "step", "spot", "skin", "slim", "snap", "trap", "trip", "drop",
        "drum", "flag", "frog", "glad", "grab", "grim", "plan", "plot", "slip", "spin", "stem", "swim", "twin", "black",
        "brick", "stick", "trick", "think", "thank", "shop", "ship", "chip", "chop", "shut", "thin", "then", "than", "that",
        "this", "them", "with", "wish", "dish", "fish", "rush", "lush", "hush", "push", "bush", "milk", "silk", "band",
        "bank", "pink", "sink", "link", "drink", "stink", "stamp", "stump", "jump", "lamp", "damp", "camp", "chimp", "limp",
        "hand", "land", "sand", "send", "bend", "lend", "mend", "tend", "wind", "kind", "find", "bind", "mind", "fund",
        "long", "song", "king", "ring", "sing", "wing", "thing", "sting", "swing", "bring", "string", "lift", "gift",
        "soft", "loft", "melt", "belt", "felt", "salt", "bolt", "halt", "kilt", "tilt", "wilt", "film", "helm", "palm",
        "calm", "real", "free", "tree", "bee", "see", "fee", "pool", "food", "mood", "good", "wood", "book", "cook", "look",
        "rain", "pain", "main", "gain", "train", "brain", "drain", "plain", "chain", "out", "about", "boat", "coat", "goat",
        "road", "load", "toad", "soap", "moon", "soon", "spoon", "noon", "boot", "root", "shoot", "seed", "need", "feed",
        "keep", "deep", "sleep", "sweep", "creep", "steep", "sheep", "peek", "seek", "week", "meek", "greek", "cheek",
        "shoe", "toe", "doe", "foe", "hoe", "joe", "woe", "zoo", "too", "boo", "moo", "goo", "shampoo", "taboo",
        "butter", "letter", "better", "bitter", "litter", "sitter", "matter", "batter", "platter", "spatter", "chatter",
        "kitten", "mitten", "bottle", "little", "middle", "riddle", "fiddle", "puddle", "cuddle", "muddle", "huddle",
        "jungle", "bundle", "candle", "handle", "sandle", "kindle", "spindle", "swindle", "dwindle", "tumble", "rumble",
        "bumble", "humble", "fumble", "grumble", "crumble", "stumble", "jumble", "mumble", "nimble", "thimble", "simple",
        "temple", "sample", "ample", "trample", "dimple", "pimple", "ripple", "tipple", "nipple", "triple", "staple",
        "baker", "maker", "taker", "sink", "wink", "mink", "rink", "kink", "blink", "clink", "brink", "shrink",
        "plank", "blank", "crank", "drank", "frank", "prank", "spank", "stank", "swank", "tank", "yank", "sank", "rank",
        "robot", "lemon", "melon", "salmon", "dragon", "wagon", "button", "cotton", "mutton", "glutton", "piston",
    ];
}
