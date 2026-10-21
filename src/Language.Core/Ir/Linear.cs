using System.Text;

namespace Language.Core.Ir;

/// <summary>
/// Reads and writes the linearized IR form that the neural model sees:
/// <c>(assert (e1 hug :tense fut :pol neg :subj (m1 Maria :propn) :obj (m2 it :pron :ref m3)))</c>.
/// The format round-trips exactly: Write(Parse(s)) == s for canonical strings.
/// </summary>
public static class Linear
{
    public static string Write(Node node)
    {
        var sb = new StringBuilder();
        Write(node, sb);
        return sb.ToString();
    }

    static void Write(Node node, StringBuilder sb)
    {
        sb.Append('(');
        if (node.Id is not null) sb.Append(node.Id).Append(' ');
        sb.Append(Quote(node.Head));
        foreach (var p in node.Props)
        {
            sb.Append(' ');
            if (p.Role != "") sb.Append(':').Append(p.Role);
            if (p.Atom is not null) sb.Append(' ').Append(Quote(p.Atom));
            else if (p.Node is not null)
            {
                if (p.Role != "") sb.Append(' ');
                Write(p.Node, sb);
            }
        }
        sb.Append(')');
    }

    static string Quote(string s) =>
        s.Length > 0 && s[0] != ':' && s.All(c => !char.IsWhiteSpace(c) && c != '(' && c != ')' && c != '"')
            ? s
            : "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

    public static Node Parse(string text)
    {
        var tokens = Tokenize(text);
        var pos = 0;
        var node = ParseNode(tokens, ref pos);
        if (pos != tokens.Count) throw new IrFormatException($"Unexpected trailing input at token {pos}: '{tokens[pos].Text}'");
        return node;
    }

    public static bool TryParse(string text, out Node? node, out string? error)
    {
        try
        {
            node = Parse(text);
            error = null;
            return true;
        }
        catch (IrFormatException e)
        {
            node = null;
            error = e.Message;
            return false;
        }
    }

    record struct Tok(string Text, bool Quoted);

    static List<Tok> Tokenize(string s)
    {
        var list = new List<Tok>();
        var i = 0;
        while (i < s.Length)
        {
            var c = s[i];
            if (char.IsWhiteSpace(c)) { i++; continue; }
            if (c == '(' || c == ')') { list.Add(new Tok(c.ToString(), false)); i++; continue; }
            if (c == '"')
            {
                var sb = new StringBuilder();
                i++;
                while (i < s.Length && s[i] != '"')
                {
                    if (s[i] == '\\' && i + 1 < s.Length) i++;
                    sb.Append(s[i++]);
                }
                if (i >= s.Length) throw new IrFormatException("Unterminated quoted atom");
                i++;
                list.Add(new Tok(sb.ToString(), true));
                continue;
            }
            var start = i;
            while (i < s.Length && !char.IsWhiteSpace(s[i]) && s[i] != '(' && s[i] != ')' && s[i] != '"') i++;
            list.Add(new Tok(s[start..i], false));
        }
        return list;
    }

    static Node ParseNode(List<Tok> t, ref int pos)
    {
        Expect(t, pos, "(");
        pos++;
        if (pos >= t.Count || IsSyntax(t[pos])) throw new IrFormatException($"Expected head at token {pos}");
        string? id = null;
        if (!t[pos].Quoted && Node.IdPattern.IsMatch(t[pos].Text) && pos + 1 < t.Count && !IsSyntax(t[pos + 1]))
            id = t[pos++].Text;
        var node = new Node(t[pos++].Text, id);
        while (true)
        {
            if (pos >= t.Count) throw new IrFormatException("Unbalanced parentheses: missing ')'");
            var tok = t[pos];
            if (!tok.Quoted && tok.Text == ")") { pos++; return node; }
            if (!tok.Quoted && tok.Text == "(") { node.Child(ParseNode(t, ref pos)); continue; }
            if (!tok.Quoted && tok.Text.StartsWith(':') && tok.Text.Length > 1)
            {
                var role = tok.Text[1..];
                pos++;
                if (pos >= t.Count) throw new IrFormatException("Role at end of input");
                var next = t[pos];
                if (!next.Quoted && next.Text == "(") node.Add(role, ParseNode(t, ref pos));
                else if (!next.Quoted && (next.Text == ")" || next.Text.StartsWith(':'))) node.Flag(role);
                else { node.Add(role, next.Text); pos++; }
                continue;
            }
            throw new IrFormatException($"Unexpected atom '{tok.Text}' at token {pos} (atoms must follow a :role)");
        }
    }

    static bool IsSyntax(Tok t) => !t.Quoted && (t.Text == "(" || t.Text == ")" || t.Text.StartsWith(':'));

    static void Expect(List<Tok> t, int pos, string s)
    {
        if (pos >= t.Count || t[pos].Quoted || t[pos].Text != s)
            throw new IrFormatException($"Expected '{s}' at token {pos}");
    }
}

public sealed class IrFormatException(string message) : Exception(message);
