using Language.Core.Ir;
using Language.Core.Text;
using Microsoft.Data.Sqlite;

namespace Language.Lexicon;

/// <summary>How a word's meaning entered the lexicon. Ordered from strongest to weakest evidence.</summary>
public enum LexStatus
{
    /// <summary>The speaker asserted the definition directly: "Snuggletronics means technologically enhanced cuddling."</summary>
    ExplicitDefinition,
    /// <summary>The speaker offered a definition with a hedge: "I think a gromp is a kind of bird."</summary>
    HedgedDefinition,
    /// <summary>The speaker reported someone else's definition: "Tom said that florp means soup."</summary>
    ReportedDefinition,
    /// <summary>The word was only used, never defined. Grammatical features (part of speech) are recorded; no meaning is.</summary>
    InferredCandidate,
}

public sealed record Sense(long Id, string Lemma, string Pos, string? DefinitionText, string? DefinitionIr, LexStatus Status,
    string SourceText, string? Speaker, DateTime Timestamp);

public sealed record LexEntry(string Lemma, string Pos, LexStatus BestStatus, IReadOnlyList<Sense> Senses);

/// <summary>
/// The engine's persistent vocabulary memory (SQLite). New words live here, never in model weights.
/// The store records who said what. It never decides whether a definition is true; that is up to external systems.
/// </summary>
public sealed class LexiconStore : IDisposable
{
    readonly SqliteConnection _db;

    public LexiconStore(string path)
    {
        _db = new SqliteConnection($"Data Source={path}");
        _db.Open();
        Exec("""
            CREATE TABLE IF NOT EXISTS lexeme (
                lemma TEXT NOT NULL, pos TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY (lemma, pos));
            CREATE TABLE IF NOT EXISTS sense (
                id INTEGER PRIMARY KEY AUTOINCREMENT, lemma TEXT NOT NULL, pos TEXT NOT NULL,
                definition_text TEXT, definition_ir TEXT, status TEXT NOT NULL,
                source_text TEXT NOT NULL, speaker TEXT, timestamp TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT, lemma TEXT NOT NULL, pos TEXT NOT NULL,
                utterance TEXT NOT NULL, ir TEXT, speaker TEXT, timestamp TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS sense_lemma ON sense(lemma);
            """);
    }

    public void Dispose() => _db.Dispose();

    void Exec(string sql, params (string, object?)[] args)
    {
        using var cmd = _db.CreateCommand();
        cmd.CommandText = sql;
        foreach (var (k, v) in args) cmd.Parameters.AddWithValue(k, v ?? DBNull.Value);
        cmd.ExecuteNonQuery();
    }

    static string Now() => DateTime.UtcNow.ToString("O");

    void EnsureLexeme(string lemma, string pos) =>
        Exec("INSERT OR IGNORE INTO lexeme(lemma,pos,created_at) VALUES($l,$p,$t)", ("$l", lemma), ("$p", pos), ("$t", Now()));

    public long AddSense(string lemma, string pos, string? definitionText, string? definitionIr, LexStatus status, string source, string? speaker)
    {
        lemma = Normalize(lemma);
        EnsureLexeme(lemma, pos);
        using var cmd = _db.CreateCommand();
        cmd.CommandText = """
            INSERT INTO sense(lemma,pos,definition_text,definition_ir,status,source_text,speaker,timestamp)
            VALUES($l,$p,$dt,$di,$s,$src,$sp,$t); SELECT last_insert_rowid();
            """;
        cmd.Parameters.AddWithValue("$l", lemma);
        cmd.Parameters.AddWithValue("$p", pos);
        cmd.Parameters.AddWithValue("$dt", (object?)definitionText ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$di", (object?)definitionIr ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$s", status.ToString());
        cmd.Parameters.AddWithValue("$src", source);
        cmd.Parameters.AddWithValue("$sp", (object?)speaker ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$t", Now());
        return (long)cmd.ExecuteScalar()!;
    }

    public void AddUsage(string lemma, string pos, string utterance, string? ir, string? speaker)
    {
        lemma = Normalize(lemma);
        EnsureLexeme(lemma, pos);
        Exec("INSERT INTO usage(lemma,pos,utterance,ir,speaker,timestamp) VALUES($l,$p,$u,$i,$s,$t)",
            ("$l", lemma), ("$p", pos), ("$u", utterance), ("$i", ir), ("$s", speaker), ("$t", Now()));
    }

    static string Normalize(string lemma) => lemma.Length > 1 && lemma.Skip(1).All(char.IsLower) ? lemma.ToLowerInvariant() : lemma;

    public IReadOnlyList<Sense> Senses(string lemma)
    {
        using var cmd = _db.CreateCommand();
        cmd.CommandText = "SELECT id,lemma,pos,definition_text,definition_ir,status,source_text,speaker,timestamp FROM sense WHERE lemma=$l ORDER BY id";
        cmd.Parameters.AddWithValue("$l", Normalize(lemma));
        return ReadSenses(cmd);
    }

    public IReadOnlyList<Sense> AllSenses()
    {
        using var cmd = _db.CreateCommand();
        cmd.CommandText = "SELECT id,lemma,pos,definition_text,definition_ir,status,source_text,speaker,timestamp FROM sense ORDER BY id";
        return ReadSenses(cmd);
    }

    static List<Sense> ReadSenses(SqliteCommand cmd)
    {
        var list = new List<Sense>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
            list.Add(new Sense(r.GetInt64(0), r.GetString(1), r.GetString(2), r.IsDBNull(3) ? null : r.GetString(3),
                r.IsDBNull(4) ? null : r.GetString(4), Enum.Parse<LexStatus>(r.GetString(5)), r.GetString(6),
                r.IsDBNull(7) ? null : r.GetString(7), DateTime.Parse(r.GetString(8)).ToUniversalTime()));
        return list;
    }

    public int UsageCount(string lemma)
    {
        using var cmd = _db.CreateCommand();
        cmd.CommandText = "SELECT COUNT(*) FROM usage WHERE lemma=$l";
        cmd.Parameters.AddWithValue("$l", Normalize(lemma));
        return Convert.ToInt32(cmd.ExecuteScalar());
    }

    /// <summary>Finds the learned entry a surface token belongs to ("snuggletrons" → "snuggletron").</summary>
    public LexEntry? Lookup(string token)
    {
        foreach (var lemma in new[] { token }.Concat(Morph.LemmaCandidates(token)).Distinct(StringComparer.OrdinalIgnoreCase))
        {
            var senses = Senses(lemma);
            if (senses.Count == 0) continue;
            var best = senses.Min(s => s.Status);
            return new LexEntry(senses[0].Lemma, senses.First(s => s.Status == best).Pos, best, senses);
        }
        return null;
    }

    public bool IsLearned(string token) => Lookup(token) is { BestStatus: not LexStatus.InferredCandidate };
}
