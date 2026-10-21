using Language.Core.Data;

namespace Language.Cli;

static class Data
{
    public static IEnumerable<Item> ReadItems(string path) => Jsonl.Read<Item>(path);
}
