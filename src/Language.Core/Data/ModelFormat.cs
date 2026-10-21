using Language.Core.Ir;

namespace Language.Core.Data;

/// <summary>The exact input strings the neural model is trained on. Shared by data generation and inference.</summary>
public static class ModelFormat
{
    public const string ParsePrefix = "parse: ";
    public const string RealizePrefix = "realize: ";
    public const string HintSeparator = " || learned: ";

    /// <summary>Model input for reading, with optional learned-word hints from the lexicon.</summary>
    public static string ParseInput(string text, IEnumerable<(string Word, string Pos)>? hints = null)
    {
        var h = hints?.ToList();
        return h is { Count: > 0 }
            ? ParsePrefix + text + HintSeparator + string.Join(", ", h.Select(x => $"{x.Word}({x.Pos})"))
            : ParsePrefix + text;
    }

    public static string RealizeInput(Node ir) => RealizePrefix + Linear.Write(ir);
}
