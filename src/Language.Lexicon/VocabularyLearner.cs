using Language.Core.Ir;
using Language.Core.Realize;
using Language.Core.Text;

namespace Language.Lexicon;

/// <summary>What happened to the lexicon after reading one utterance.</summary>
public sealed record LearnReport(List<string> Unknown, List<Sense> Stored, List<string> Recognized);

/// <summary>
/// Connects the reader's IR to the persistent lexicon:
/// marks unfamiliar and learned words, stores definitions with provenance and an evidence status,
/// supplies learned-word hints for the model, and chooses learned terms when writing.
/// </summary>
public sealed class VocabularyLearner(LexiconStore store)
{
    static readonly HashSet<string> HedgeVerbs = ["think", "believe", "guess", "suppose", "suspect", "assume", "feel", "hope", "imagine"];

    /// <summary>Learned words in the text, for the model's input hints ("|| learned: snuggletronics(n)").</summary>
    public List<(string Word, string Pos)> Hints(string text)
    {
        var hints = new List<(string, string)>();
        foreach (var t in Tokenizer.Tokenize(text))
        {
            if (Words.IsPunctuation(t.Text)) continue;
            if (store.Lookup(t.Text) is { BestStatus: not LexStatus.InferredCandidate } e && !hints.Any(h => h.Item1 == t.Text))
                hints.Add((t.Text, e.Pos));
        }
        return hints;
    }

    /// <summary>
    /// Annotates the IR with :unk (not a known English word and never learned) and :lex (learned, with status),
    /// stores definitions and records usage of unfamiliar words. Returns what changed.
    /// </summary>
    public LearnReport Process(Node ir, string text, string? speaker = null)
    {
        var report = new LearnReport(new(), new(), new());

        // 1. Definitions: store them before annotating, so the defined word is recognized in the same pass.
        foreach (var (def, status) in FindDefinitions(ir))
        {
            var term = def.Get("term")!;
            var body = def.Get("def")!;
            var pos = term.IsEvent ? "v" : "n";
            var bodyIr = Linear.Write(Schema.Canonicalize(StripIds(body)));
            var bodyText = DefinitionText(body);
            var id = store.AddSense(term.Head, pos, bodyText, bodyIr, status, text, speaker);
            report.Stored.Add(store.Senses(term.Head).First(s => s.Id == id));
        }

        // 2. Mark every content head.
        foreach (var (node, _, parent, role) in ir.Walk())
        {
            if (parent is null || (node.IsCoordination || node.Head == "define")) continue;
            if (node.Has("pron") || node.Has("wh") || node.Has("implicit") || node.Has("propn")) continue;
            node.Remove("unk");
            node.Remove("lex");
            var entry = store.Lookup(node.Head);
            if (entry is not null && entry.BestStatus != LexStatus.InferredCandidate)
            {
                node.Add("lex", Status(entry.BestStatus));
                report.Recognized.Add(node.Head);
                store.AddUsage(entry.Lemma, entry.Pos, text, Linear.Write(ir), speaker);
            }
            else if (!CoreWords.IsKnown(node.Head))
            {
                node.Flag("unk");
                report.Unknown.Add(node.Head);
                var pos = node.IsEvent ? "v" : node.IsMention ? "n" : role == "adv" ? "adv" : "adj";
                if (!IsDefinedHere(ir, node))
                {
                    if (entry is null)
                    {
                        var id = store.AddSense(node.Head, pos, null, null, LexStatus.InferredCandidate, text, speaker);
                        report.Stored.Add(store.Senses(node.Head).First(s => s.Id == id));
                    }
                    store.AddUsage(node.Head, pos, text, Linear.Write(ir), speaker);
                }
            }
        }
        return report;
    }

    static string Status(LexStatus s) => s switch
    {
        LexStatus.ExplicitDefinition => "explicit",
        LexStatus.HedgedDefinition => "hedged",
        LexStatus.ReportedDefinition => "reported",
        _ => "inferred",
    };

    static bool IsDefinedHere(Node ir, Node node) =>
        ir.Descendants().Any(d => d.Id is null && d.Head == "define" && ReferenceEquals(d.Get("term"), node));

    /// <summary>Finds define nodes and classifies the evidence: a bare definition is explicit; one under the
    /// speaker's own "I think" is hedged; one under someone else's "said/thinks" is reported.
    /// Negated or modal framings ("X might mean Y") count as hedged.</summary>
    public static IEnumerable<(Node Define, LexStatus Status)> FindDefinitions(Node root)
    {
        if (root.Head == "define")
        {
            yield return (root, LexStatus.ExplicitDefinition);
            yield break;
        }
        foreach (var (node, _, parent, role) in root.Walk())
        {
            if (node.Head != "define" || node.Id is not null || parent is null) continue;
            if (role != "comp" || !parent.IsEvent) continue;
            var subj = parent.Get("subj");
            var speakerIsSubject = subj is not null && subj.Has("pron") && subj.Head is "I" or "we";
            var hedged = parent.Has("mod") || parent.Atom("pol") == "neg" || HedgeVerbs.Contains(parent.Head);
            if (parent.Atom("pol") == "neg") continue; // "I don't think X means Y" asserts no definition
            yield return (node, speakerIsSubject ? (hedged ? LexStatus.HedgedDefinition : LexStatus.ExplicitDefinition) : LexStatus.ReportedDefinition);
        }
    }

    static Node StripIds(Node n)
    {
        var c = Schema.StripRuntime(n);
        foreach (var d in c.Descendants()) d.Id = null;
        return c;
    }

    static string DefinitionText(Node body) => Realizer.Phrase(body);

    /// <summary>
    /// Lexical choice for writing: when an IR node has exactly the structure of a stored definition
    /// (same head, and all the definition's modifiers), replace it with the learned term.
    /// Only definitions whose status is at least <paramref name="minimum"/> are used.
    /// </summary>
    public int ApplyLearnedTerms(Node ir, LexStatus minimum = LexStatus.ExplicitDefinition)
    {
        var senses = store.AllSenses().Where(s => s.DefinitionIr is not null && s.Status <= minimum).ToList();
        var replaced = 0;
        foreach (var node in ir.Descendants().ToList())
        {
            if (node.Id is null) continue;
            foreach (var s in senses)
            {
                Node def;
                try { def = Linear.Parse(s.DefinitionIr!); } catch (IrFormatException) { continue; }
                if ((s.Pos == "v") != node.IsEvent || def.Head != node.Head) continue;
                // Determiners and number frame the phrase; they are not part of the meaning being named.
                var nodeProps = node.Props.Select(PropKey).ToList();
                var defProps = def.Props.Where(p => p.Role is not ("det" or "num")).Select(PropKey).ToList();
                if (!defProps.All(nodeProps.Contains)) continue;
                node.Head = s.Lemma;
                node.Props.RemoveAll(p => defProps.Contains(PropKey(p)));
                replaced++;
                break;
            }
        }
        return replaced;
    }

    static string PropKey(Prop p)
    {
        if (p.Node is null) return $":{p.Role} {p.Atom}";
        var c = p.Node.Clone();
        foreach (var d in c.Descendants()) d.Id = null;
        return $":{p.Role} {Linear.Write(Schema.Canonicalize(c))}";
    }
}
