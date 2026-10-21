using System.Text.Json;
using System.Text.Json.Nodes;

namespace Language.Core.Ir;

/// <summary>
/// JSON view of the IR for external systems. Each node becomes an object with "id" (optional), "head",
/// one key per role (an array when the role repeats), and "args" for positional children.
/// Flags become <c>true</c>. Conversion is lossless in both directions.
/// </summary>
public static class IrJson
{
    static readonly JsonSerializerOptions Pretty = new() { WriteIndented = true };

    public static string Serialize(Node node, bool indented = true) =>
        ToJson(node).ToJsonString(indented ? Pretty : new JsonSerializerOptions());

    public static JsonObject ToJson(Node node)
    {
        var o = new JsonObject();
        if (node.Id is not null) o["id"] = node.Id;
        o["head"] = node.Head;
        foreach (var group in node.Props.GroupBy(p => p.Role))
        {
            var key = group.Key == "" ? "args" : group.Key;
            var values = group.Select(Value).ToList();
            o[key] = values.Count == 1 && group.Key != "" ? values[0] : new JsonArray(values.ToArray());
        }
        return o;
    }

    static JsonNode? Value(Prop p) =>
        p.Node is not null ? ToJson(p.Node) : p.Atom is not null ? JsonValue.Create(p.Atom) : JsonValue.Create(true);

    public static Node Deserialize(string json) =>
        FromJson(JsonNode.Parse(json)?.AsObject() ?? throw new IrFormatException("Empty JSON"));

    public static Node FromJson(JsonObject o)
    {
        var head = o["head"]?.GetValue<string>() ?? throw new IrFormatException("Node without \"head\"");
        var node = new Node(head, o["id"]?.GetValue<string>());
        foreach (var (key, value) in o)
        {
            if (key is "id" or "head" || value is null) continue;
            var role = key == "args" ? "" : key;
            var items = value is JsonArray arr ? arr.ToList() : [value];
            foreach (var item in items)
            {
                switch (item)
                {
                    case JsonObject child: node.Add(role, FromJson(child)); break;
                    case JsonValue v when v.GetValueKind() == JsonValueKind.True: node.Flag(role); break;
                    case JsonValue v: node.Add(role, v.ToString()); break;
                    default: throw new IrFormatException($"Unsupported JSON value for role '{key}'");
                }
            }
        }
        return node;
    }
}
