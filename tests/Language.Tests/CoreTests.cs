using Language.Core.Ir;
using Language.Core.Realize;
using Language.Core.Text;

namespace Language.Tests;

public class LinearTests
{
    [Theory]
    [InlineData("(assert (e1 hug :tense fut :pol neg :subj (m1 Maria :propn) :obj (m2 it :pron :ref m3)))")]
    [InlineData("(define :term (m1 snuggletronics) :def (m2 cuddling :adj (enhanced :adv technologically)) :marker means)")]
    [InlineData("(assert :ambig attachment (and (e1 rain :tense past :subj (m1 it :pron :ref none)) (e2 stay :tense past :subj (m2 we :pron))))")]
    public void RoundTripsExactly(string ir) => Assert.Equal(ir, Linear.Write(Linear.Parse(ir)));

    [Fact]
    public void QuotesAtomsThatNeedIt()
    {
        var n = new Node("assert").Child(new Node("go", "e1").Add("tense", "past").Add("adv", "o'clock now"));
        var text = Linear.Write(n);
        Assert.Contains("\"o'clock now\"", text);
        Assert.Equal(text, Linear.Write(Linear.Parse(text)));
    }

    [Theory]
    [InlineData("(assert (e1 go")]
    [InlineData("(assert (e1 go :tense past)) extra")]
    [InlineData("(assert stray)")]
    public void RejectsMalformedInput(string bad) => Assert.False(Linear.TryParse(bad, out _, out _));

    [Fact]
    public void JsonViewIsLossless()
    {
        var ir = Linear.Parse("(assert (e1 give :tense past :subj (m1 Tom :propn) :iobj (m2 I :pron) :obj (m3 book :det a :adj old :adj red)))");
        Assert.Equal(Linear.Write(ir), Linear.Write(IrJson.Deserialize(IrJson.Serialize(ir))));
    }
}

public class ValidatorTests
{
    [Fact]
    public void AcceptsWellFormedIr() =>
        Assert.Empty(Validator.Validate(Linear.Parse("(ynq (e1 open :mod can :subj (m1 you :pron) :obj (m2 door :det the)))")));

    [Theory]
    [InlineData("(assert (e1 sleep :subj (m1 dog :det the)))", "needs :tense")]
    [InlineData("(assert (e1 see :tense past :subj (m1 he :pron :ref m9)))", "unknown mention")]
    [InlineData("(command (e1 sleep :tense pres :subj (m1 you :pron :implicit)))", "must not have :tense")]
    [InlineData("(whq (e1 sleep :tense past :subj (m1 dog :det the)))", "whq needs")]
    [InlineData("(assert (e1 sleep :tense yesterday :subj (m1 dog)))", ":tense must be")]
    public void ReportsErrors(string ir, string expected) =>
        Assert.Contains(Validator.Validate(Linear.Parse(ir)), e => e.Contains(expected));

    [Fact]
    public void RelativeClauseGapMustReferToHost()
    {
        var ok = Linear.Parse("(assert (e1 sleep :tense past :subj (m1 man :det the :rel (e2 see :tense past :subj (m2 I :pron) :obj m1))))");
        Assert.Empty(Validator.Validate(ok));
        var bad = Linear.Parse("(assert (e1 sleep :tense past :subj (m1 man :det the) :obj m1))");
        Assert.NotEmpty(Validator.Validate(bad));
    }
}

public class MorphTests
{
    [Theory]
    [InlineData("stop", "stopped", "stopping", "stops")]
    [InlineData("hug", "hugged", "hugging", "hugs")]
    [InlineData("visit", "visited", "visiting", "visits")]
    [InlineData("carry", "carried", "carrying", "carries")]
    [InlineData("make", "made", "making", "makes")]
    [InlineData("oversleep", "overslept", "oversleeping", "oversleeps")]
    [InlineData("gorp", "gorped", "gorping", "gorps")]
    [InlineData("zindle", "zindled", "zindling", "zindles")]
    public void InflectsVerbs(string lemma, string past, string ing, string s)
    {
        Assert.Equal(past, Morph.Past(lemma));
        Assert.Equal(ing, Morph.PresentParticiple(lemma));
        Assert.Equal(s, Morph.ThirdSingular(lemma));
    }

    [Theory]
    [InlineData("child", "children")]
    [InlineData("box", "boxes")]
    [InlineData("city", "cities")]
    [InlineData("snuggletron", "snuggletrons")]
    public void Pluralizes(string sg, string pl) => Assert.Equal(pl, Morph.Plural(sg));

    [Theory]
    [InlineData("cancel", "cancelled")]
    [InlineData("go", "went")]
    [InlineData("he", "him")]
    [InlineData("I", "my")]
    [InlineData("a", "an")]
    [InlineData("not", "n't")]
    public void AnchorsSurfaceForms(string lemma, string form) => Assert.True(Morph.CanSurfaceAs(lemma, form));

    [Fact]
    public void TokenizerSplitsClitics()
    {
        var t = Tokenizer.Tokenize("I'm sure Maria won't hug it.").Select(x => x.Text).ToList();
        Assert.Equal(["I", "'m", "sure", "Maria", "wo", "n't", "hug", "it", "."], t);
    }
}

public class RealizerTests
{
    [Theory]
    [InlineData("(assert (e2 hug :tense fut :pol neg :subj (m2 Maria :propn) :obj (m3 it :pron :ref m1) :condition (e1 charge :tense pres :pol neg :subj (m1 snuggletron :det the))))",
        "Maria will not hug it if the snuggletron does not charge.")]
    [InlineData("(ynq (e1 sleep :tense past :aspect prog :pol neg :subj (m1 cat :det a :adj black)))", "Was a black cat not sleeping?")]
    [InlineData("(whq (e1 eat :tense past :subj (m1 she :pron) :obj (m2 what :wh)))", "What did she eat?")]
    [InlineData("(whq (e1 talk :tense past :subj (m1 you :pron) :obl-to (m2 who :wh) :obl-at (m3 party :det the)))", "Who did you talk to at the party?")]
    [InlineData("(command (e1 open :pol neg :subj (m1 you :pron :implicit) :obj (m2 window :det the)))", "Do not open the window.")]
    [InlineData("(command (e1 call :subj (m1 you :pron :implicit) :obj (m2 grandmother :poss (m3 you :pron)) :adv please :adv tomorrow))", "Please call your grandmother tomorrow.")]
    [InlineData("(assert (e1 sleep :tense pres :pol neg :negword never :subj (m1 she :pron)))", "She never sleeps.")]
    [InlineData("(assert (e1 sleep :tense past :aspect perfprog :pol neg :subj (m1 she :pron)))", "She had not been sleeping.")]
    [InlineData("(assert (e1 eat :tense past :voice pass :subj (m1 cake :det the) :obl-by (m2 dog :det the)))", "The cake was eaten by the dog.")]
    [InlineData("(assert (e1 want :tense pres :subj (m1 child :det the :num pl) :xcomp (e2 play :obl-in (m2 garden :det the))))", "The children want to play in the garden.")]
    [InlineData("(define :term (m1 snuggletronics) :def (m2 cuddling :adj (enhanced :adv technologically)) :marker means)", "Snuggletronics means technologically enhanced cuddling.")]
    [InlineData("(assert (e1 be :tense pres :there :subj (m1 cat :det a) :obl-on (m2 mat :det the)))", "There is a cat on the mat.")]
    [InlineData("(assert (e1 wait :tense past :subj (m1 we :pron) :obl-at (m2 stop :det a :compound bus)))", "We waited at a bus stop.")]
    [InlineData("(assert (e1 say :tense past :subj (m1 woman :det the :rel (e2 sell :tense past :subj m1 :iobj (m3 I :pron) :obj (m4 bike :det the))) :comp (e3 be :tense past :subj (m5 it :pron :ref m4) :attr fast)))",
        "The woman that sold me the bike said that it was fast.")]
    public void RealizesDefaultSurface(string ir, string expected) => Assert.Equal(expected, Realizer.Realize(Linear.Parse(ir)));

    [Fact]
    public void ContractionsAreASurfaceOption()
    {
        var ir = Linear.Parse("(assert (e1 come :tense fut :pol neg :subj (m1 I :pron)))");
        Assert.Equal("I won't come.", Realizer.Realize(ir, new RealizerOptions { Contractions = true }));
    }

    [Fact]
    public void FrontedConditionKeepsMeaning()
    {
        var ir = Linear.Parse("(assert (e1 stay :tense fut :subj (m1 we :pron) :condition (e2 rain :tense pres :subj (m2 it :pron :ref none))))");
        Assert.Equal("If it rains, we will stay.", Realizer.Realize(ir, new RealizerOptions { FrontSubordinate = true }));
    }
}

public class AnchoringTests
{
    [Fact]
    public void FlagsInventedContent()
    {
        var text = "The dog slept.";
        var invented = Linear.Parse("(assert (e1 sleep :tense past :subj (m1 dog :det the :adj happy) :obl-in (m2 park :det the)))");
        var a = Anchoring.Compute(invented, Tokenizer.Tokenize(text));
        Assert.Contains(a.Unanchored, u => u.Contains("happy"));
        Assert.Contains(a.Unanchored, u => u.Contains("park"));
    }

    [Fact]
    public void ImplicitSubjectIsNotInvented()
    {
        var ir = Linear.Parse("(command (e1 sit :subj (m1 you :pron :implicit)))");
        Assert.Empty(Anchoring.Compute(ir, Tokenizer.Tokenize("Sit!")).Unanchored);
    }

    [Fact]
    public void CoverageFindsOmittedWords()
    {
        var ir = Linear.Parse("(assert (e1 sleep :tense past :subj (m1 dog :det the)))");
        Assert.Equal(["happily"], Coverage.Uncovered(ir, Tokenizer.Tokenize("The dog slept happily.")).ToList());
    }
}
