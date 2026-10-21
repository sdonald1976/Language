using Language.Core.Data;
using Language.Core.Ir;
using Language.Core.Realize;
using Language.Core.Text;
using Language.DataGen;
using Language.Eval;
using Language.Lexicon;

namespace Language.Tests;

public class EvalTests
{
    const string Text = "Maria won't hug it if the snuggletron doesn't charge.";
    const string Gold = "(assert (e1 hug :tense fut :pol neg :subj (m1 Maria :propn) :obj (m2 it :pron :ref m3) :condition (e2 charge :tense pres :pol neg :subj (m3 snuggletron :det the))))";

    static ParseScores Score(string pred)
    {
        var item = new Item { Id = "x", Category = "T", Text = Text, Ir = Gold };
        return ParseEvaluator.Evaluate([item], new Dictionary<string, string?> { ["x"] = pred }).Slices["all"];
    }

    [Fact]
    public void PerfectPredictionScoresOneEvenWithDifferentIds()
    {
        var renumbered = "(assert (e7 hug :tense fut :pol neg :subj (m4 Maria :propn) :obj (m5 it :pron :ref m6) :condition (e8 charge :tense pres :pol neg :subj (m6 snuggletron :det the))))";
        var s = Score(renumbered);
        Assert.Equal(1.0, s.Triples.F1, 6);
        Assert.Equal(1.0, s.Exact.Value);
        Assert.Equal(1.0, s.CorefResolvable.Value);
    }

    [Fact]
    public void WrongTenseAndInventionAreCounted()
    {
        var s = Score("(assert (e1 hug :tense past :pol neg :subj (m1 Maria :propn :adj happy) :obj (m2 it :pron :ref m3) :condition (e2 charge :tense pres :pol neg :subj (m3 snuggletron :det the))))");
        Assert.True(s.Triples.F1 < 1);
        Assert.Equal(1, s.InventedItems);
        Assert.True(s.Features["tense"].Value < 1);
    }

    [Fact]
    public void MalformedPredictionGetsZero()
    {
        var s = Score("(assert (e1 hug");
        Assert.Equal(0, s.Triples.F1);
        Assert.Equal(0, s.WellFormed.Value);
    }

    [Fact]
    public void WritingAdditionsAndOmissionsAreDetected()
    {
        var item = new Item { Id = "w", Category = "T11", Ir = "(assert (e1 bark :tense past :subj (m1 cat :det the)))", Forbidden = ["meowed"] };
        var outputs = new Dictionary<string, string?> { ["w"] = "Sadly, the cat meowed." };
        var (slices, items) = WriteEvaluator.Evaluate([item], outputs);
        Assert.Equal(1.0, slices["all"].HasAddition.Value);
        Assert.Contains("Sadly", items[0].Additions);
        Assert.Contains("meowed", items[0].Forbidden);
        Assert.NotEmpty(items[0].Omissions); // "bark" is missing
    }
}

public class LexiconTests : IDisposable
{
    readonly string _db = Path.Combine(Path.GetTempPath(), $"lex-{Guid.NewGuid():N}.db");
    readonly LexiconStore _store;
    readonly VocabularyLearner _learner;

    public LexiconTests()
    {
        _store = new LexiconStore(_db);
        _learner = new VocabularyLearner(_store);
    }

    public void Dispose()
    {
        _store.Dispose();
        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        File.Delete(_db);
    }

    LearnReport Read(string text, string ir, string? speaker = null) => _learner.Process(Linear.Parse(ir), text, speaker);

    [Fact]
    public void ExplicitDefinitionIsStoredWithProvenanceAndRecognizedLater()
    {
        var r = Read("Snuggletronics means technologically enhanced cuddling.",
            "(define :term (m1 snuggletronics) :def (m2 cuddling :adj (enhanced :adv technologically)) :marker means)", "Ava-user");
        var s = Assert.Single(r.Stored);
        Assert.Equal(LexStatus.ExplicitDefinition, s.Status);
        Assert.Equal("technologically enhanced cuddling", s.DefinitionText);
        Assert.Equal("Ava-user", s.Speaker);

        var later = Read("We tried snuggletronics.", "(assert (e1 try :tense past :subj (m1 we :pron) :obj (m2 snuggletronics)))");
        Assert.Contains("snuggletronics", later.Recognized);
        Assert.Empty(later.Stored);
    }

    [Fact]
    public void HedgedReportedAndNegatedDefinitionsAreDistinguished()
    {
        Read("I think a gromp is a kind of bird.", "(assert (e1 think :tense pres :subj (m1 I :pron) :comp (define :term (m2 gromp :det a) :def (m3 kind :det a :obl-of (m4 bird)) :marker is)))");
        Read("Tom said that florp means soup.", "(assert (e1 say :tense past :subj (m1 Tom :propn) :comp (define :term (m2 florp) :def (m3 soup) :marker means)))");
        var neg = Read("I don't think wug means cat.", "(assert (e1 think :tense pres :pol neg :subj (m1 I :pron) :comp (define :term (m2 wug) :def (m3 cat) :marker means)))");
        Assert.Equal(LexStatus.HedgedDefinition, _store.Lookup("gromp")!.BestStatus);
        Assert.Equal(LexStatus.ReportedDefinition, _store.Lookup("florp")!.BestStatus);
        Assert.DoesNotContain(neg.Stored, s => s.Status != LexStatus.InferredCandidate);
    }

    [Fact]
    public void MereUseStoresNoMeaning()
    {
        var r = Read("I fed the zibbet.", "(assert (e1 feed :tense past :subj (m1 I :pron) :obj (m2 zibbet :det the)))");
        Assert.Contains("zibbet", r.Unknown);
        var s = Assert.Single(r.Stored);
        Assert.Equal(LexStatus.InferredCandidate, s.Status);
        Assert.Null(s.DefinitionText);
        Assert.False(_store.IsLearned("zibbet"));
    }

    [Fact]
    public void InflectedFormsAreRecognized()
    {
        Read("A snuggletron is a cuddly robot.", "(define :term (m1 snuggletron :det a) :def (m2 robot :det a :adj cuddly) :marker is)");
        Assert.NotNull(_store.Lookup("snuggletrons"));
    }

    [Fact]
    public void WritingUsesOnlyExplicitTermsAndKeepsTheDeterminer()
    {
        Read("A snuggletron is a cuddly robot.", "(define :term (m1 snuggletron :det a) :def (m2 robot :det a :adj cuddly) :marker is)");
        Read("I think a gromp is a kind of bird.", "(assert (e1 think :tense pres :subj (m1 I :pron) :comp (define :term (m2 gromp :det a) :def (m3 kind :det a :obl-of (m4 bird)) :marker is)))");
        var ir = Linear.Parse("(assert (e1 buy :tense past :subj (m1 Tom :propn) :obj (m2 robot :det a :adj cuddly) :obl-for (m3 kind :det a :obl-of (m4 bird))))");
        Assert.Equal(1, _learner.ApplyLearnedTerms(ir));
        Assert.Equal("Tom bought a snuggletron for a kind of bird.", Realizer.Realize(ir));
    }
}

public class GeneratorTests
{
    [Fact]
    public void SyntheticPairsAreValidAnchoredAndCovered()
    {
        var rng = new Random(7);
        var pool = new Nonsense(Vocab.Adjectives).Pool(rng, 200);
        var sampler = new Sampler(rng, pool);
        for (var i = 0; i < 300; i++)
        {
            var ir = sampler.Sample();
            var text = Realizer.Realize(ir, RealizerOptions.Sample(rng));
            Assert.Empty(Validator.Validate(ir));
            var tokens = Tokenizer.Tokenize(text);
            Assert.Empty(Anchoring.Compute(ir, tokens).Unanchored);
            Assert.Empty(Coverage.Uncovered(ir, tokens));
        }
    }
}

public class UdConverterTests
{
    const string Conllu = """
        # sent_id = s1
        # text = The cat that I adopted sleeps on my bed.
        1	The	the	DET	DT	Definite=Def|PronType=Art	2	det	_	_
        2	cat	cat	NOUN	NN	Number=Sing	6	nsubj	_	_
        3	that	that	PRON	WDT	PronType=Rel	5	obj	_	_
        4	I	I	PRON	PRP	Case=Nom|Number=Sing|Person=1|PronType=Prs	5	nsubj	_	_
        5	adopted	adopt	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	2	acl:relcl	_	_
        6	sleeps	sleep	VERB	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	_	_
        7	on	on	ADP	IN	_	9	case	_	_
        8	my	my	PRON	PRP$	Number=Sing|Person=1|Poss=Yes|PronType=Prs	9	nmod:poss	_	_
        9	bed	bed	NOUN	NN	Number=Sing	6	obl	_	_
        10	.	.	PUNCT	.	_	6	punct	_	_

        """;

    [Fact]
    public void ConvertsRelativeClauseWithGap()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, Conllu.Replace("        ", ""));
        var s = UdConverter.ReadConllu(path).Single();
        var ir = new UdConverter(strict: true).Convert(s);
        Assert.NotNull(ir);
        Assert.Equal("(assert (e1 sleep :tense pres :subj (m1 cat :det the :rel (e2 adopt :tense past :subj (m2 I :pron) :obj m1)) :obl-on (m3 bed :poss (m4 I :pron))))", Linear.Write(ir!));
    }
}
