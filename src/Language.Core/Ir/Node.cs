using System.Text.RegularExpressions;

namespace Language.Core.Ir;

/// <summary>
/// One node of the intermediate representation (IR v0).
/// The IR is a tree: <c>(id? head :role value ...)</c>. Values are atoms or nested nodes.
/// Positional children (no role) are stored with <see cref="Prop.Role"/> == "".
/// See docs/IR-v0.md for the meaning of heads and roles.
/// </summary>
public sealed class Node
{
    public string? Id { get; set; }
    public string Head { get; set; }
    public List<Prop> Props { get; } = new();

    public Node(string head, string? id = null)
    {
        Head = head;
        Id = id;
    }

    public static readonly Regex IdPattern = new(@"^[em]\d+$", RegexOptions.Compiled);

    public bool IsEvent => Id is not null && Id[0] == 'e';
    /// <summary>A coordination ("and"/"or"/"but" with positional members). Noun coordinations may carry an id (mN).</summary>
    public bool IsCoordination => !IsEvent && Schema.Coordinators.Contains(Head) && Props.Any(p => p.Role == "" && p.Node is not null);
    public bool IsMention => Id is not null && Id[0] == 'm';

    public Node Add(string role, string atom) { Props.Add(new Prop(role, atom, null)); return this; }
    public Node Add(string role, Node node) { Props.Add(new Prop(role, null, node)); return this; }
    public Node Flag(string role) { Props.Add(new Prop(role, null, null)); return this; }
    public Node Child(Node node) { Props.Add(new Prop("", null, node)); return this; }

    public bool Has(string role) => Props.Any(p => p.Role == role);
    public string? Atom(string role) => Props.FirstOrDefault(p => p.Role == role)?.Atom;
    public Node? Get(string role) => Props.FirstOrDefault(p => p.Role == role)?.Node;
    public IEnumerable<Prop> All(string role) => Props.Where(p => p.Role == role);
    public IEnumerable<string> Atoms(string role) => All(role).Where(p => p.Atom is not null).Select(p => p.Atom!);
    public IEnumerable<Node> Children => Props.Where(p => p.Node is not null).Select(p => p.Node!);
    public IEnumerable<Node> Positional => Props.Where(p => p.Role == "" && p.Node is not null).Select(p => p.Node!);

    public void Remove(string role) => Props.RemoveAll(p => p.Role == role);

    public void Set(string role, string atom)
    {
        Remove(role);
        Add(role, atom);
    }

    /// <summary>All nodes in the subtree, depth-first, parent before children.</summary>
    public IEnumerable<Node> Descendants()
    {
        yield return this;
        foreach (var c in Children)
            foreach (var d in c.Descendants())
                yield return d;
    }

    /// <summary>Like <see cref="Descendants"/> but also yields the role path to each node.</summary>
    public IEnumerable<(Node Node, string Path, Node? Parent, string Role)> Walk(string path = "", Node? parent = null, string role = "")
    {
        yield return (this, path, parent, role);
        var i = 0;
        foreach (var p in Props)
        {
            if (p.Node is null) continue;
            var r = p.Role == "" ? $"#{i++}" : p.Role;
            foreach (var w in p.Node.Walk(path + "/" + r, this, p.Role))
                yield return w;
        }
    }

    public Node Clone()
    {
        var n = new Node(Head, Id);
        foreach (var p in Props)
            n.Props.Add(new Prop(p.Role, p.Atom, p.Node?.Clone()));
        return n;
    }

    public override string ToString() => Linear.Write(this);
}

/// <summary>A role/value pair. Exactly one of Atom/Node is set, or neither for a flag.</summary>
public sealed record Prop(string Role, string? Atom, Node? Node)
{
    public bool IsFlag => Atom is null && Node is null;
}
