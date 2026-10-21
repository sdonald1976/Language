using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace Language.Engine;

/// <summary>Settings written by training/export_onnx.py.</summary>
public sealed record ModelInfo(int DecoderStartTokenId, int EosTokenId, int PadTokenId, int VocabSize, int MaxSource, int MaxTarget,
    Dictionary<string, string> Files, double ParamsMillions);

/// <summary>
/// A T5-family encoder-decoder exported to ONNX, run from C# with beam search.
/// The decoder is exported without a key/value cache, so each step re-reads the prefix: simple and version-proof,
/// fast enough for sentence-length outputs on CPU.
/// </summary>
public sealed class Seq2SeqModel : IDisposable
{
    readonly InferenceSession _encoder;
    readonly InferenceSession _decoder;
    readonly InferenceSession? _crossKv;
    readonly InferenceSession? _step;
    readonly SentencePieceTokenizer _tokenizer;
    public ModelInfo Info { get; }
    public string Directory { get; }

    public Seq2SeqModel(string dir, int threads = 0)
    {
        Directory = dir;
        var json = File.ReadAllText(Path.Combine(dir, "engine.json"));
        Info = JsonSerializer.Deserialize<ModelInfo>(json, new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower })
               ?? throw new InvalidDataException("bad engine.json");
        var opts = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        if (threads > 0) opts.IntraOpNumThreads = threads;
        _encoder = new InferenceSession(Path.Combine(dir, Info.Files["encoder"]), opts);
        _decoder = new InferenceSession(Path.Combine(dir, Info.Files["decoder"]), opts);
        if (Info.Files.TryGetValue("decoder_step", out var stepFile) && Info.Files.TryGetValue("cross_kv", out var crossFile))
        {
            _crossKv = new InferenceSession(Path.Combine(dir, crossFile), opts);
            _step = new InferenceSession(Path.Combine(dir, stepFile), opts);
        }
        using var spm = File.OpenRead(Path.Combine(dir, "spiece.model"));
        _tokenizer = SentencePieceTokenizer.Create(spm, false, true, null);
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
        _crossKv?.Dispose();
        _step?.Dispose();
    }

    public IReadOnlyList<int> Encode(string text) => _tokenizer.EncodeToIds(text);

    public string Decode(IEnumerable<int> ids) =>
        _tokenizer.Decode(ids.Where(i => i != Info.EosTokenId && i != Info.PadTokenId)) ?? "";

    /// <summary>Returns up to <paramref name="beams"/> hypotheses, best first, with length-normalized log-probabilities.</summary>
    public List<(string Text, double Score)> Generate(string input, int beams = 4, int? maxNew = null)
    {
        var src = Encode(input).Take(Info.MaxSource).ToArray();
        if (src.Length == 0 || src[^1] != Info.EosTokenId) src = [.. src.Take(Info.MaxSource - 1), Info.EosTokenId];
        var s = src.Length;
        var ids = new DenseTensor<long>(src.Select(x => (long)x).ToArray(), [1, s]);
        var mask = new DenseTensor<long>(Enumerable.Repeat(1L, s).ToArray(), [1, s]);
        using var encOut = _encoder.Run([NamedOnnxValue.CreateFromTensor("input_ids", ids), NamedOnnxValue.CreateFromTensor("attention_mask", mask)]);
        var hidden = encOut.First().AsTensor<float>();
        var d = hidden.Dimensions[2];
        var hiddenData = hidden.ToArray();

        var limit = maxNew ?? Info.MaxTarget;
        if (_step is not null) return GenerateCached(hidden, s, d, beams, limit);
        var live = new List<(List<int> Seq, double LogP)> { ([Info.DecoderStartTokenId], 0) };
        var done = new List<(List<int> Seq, double LogP)>();
        for (var step = 0; step < limit && live.Count > 0; step++)
        {
            var k = live.Count;
            var t = live[0].Seq.Count;
            var dec = new DenseTensor<long>([k, t]);
            for (var i = 0; i < k; i++)
                for (var j = 0; j < t; j++) dec[i, j] = live[i].Seq[j];
            var hid = new DenseTensor<float>([k, s, d]);
            var hspan = hid.Buffer.Span;
            for (var i = 0; i < k; i++) hiddenData.CopyTo(hspan.Slice(i * s * d, s * d));
            var m = new DenseTensor<long>(Enumerable.Repeat(1L, k * s).ToArray(), [k, s]);
            using var outs = _decoder.Run([
                NamedOnnxValue.CreateFromTensor("decoder_input_ids", dec),
                NamedOnnxValue.CreateFromTensor("encoder_hidden", hid),
                NamedOnnxValue.CreateFromTensor("attention_mask", m)]);
            var logits = outs.First().AsTensor<float>();
            var vocab = logits.Dimensions[2];

            var candidates = new List<(List<int> Seq, double LogP)>();
            for (var i = 0; i < k; i++)
            {
                var row = new float[vocab];
                for (var v = 0; v < vocab; v++) row[v] = logits[i, t - 1, v];
                var max = row.Max();
                var logSum = max + Math.Log(row.Sum(x => Math.Exp(x - max)));
                foreach (var v in TopK(row, beams))
                    candidates.Add(([.. live[i].Seq, v], live[i].LogP + row[v] - logSum));
            }
            live.Clear();
            foreach (var c in candidates.OrderByDescending(c => c.LogP))
            {
                if (c.Seq[^1] == Info.EosTokenId) done.Add(c);
                else live.Add(c);
                if (live.Count == beams) break;
            }
            if (done.Count >= beams && done.Max(x => Norm(x)) >= live.Select(Norm).DefaultIfEmpty(double.MinValue).Max()) break;
        }
        done.AddRange(live);
        return done.OrderByDescending(Norm).Take(beams)
            .Select(x => (Decode(x.Seq.Skip(1)), Norm(x))).ToList();
    }

    /// <summary>Beam search with the KV-cached step decoder (decoder_step.onnx): one token per call per beam.</summary>
    List<(string Text, double Score)> GenerateCached(Tensor<float> hidden, int s, int d, int beams, int limit)
    {
        using var crossOut = _crossKv!.Run([NamedOnnxValue.CreateFromTensor("encoder_hidden", hidden)]);
        var cross = crossOut.First().AsTensor<float>();
        var dims = cross.Dimensions.ToArray(); // [L, 2, 1, h, s, dk]
        int layers = dims[0], heads = dims[3], dk = dims[5];
        var crossData = cross.ToArray();
        var blockCross = heads * s * dk; // one (layer, k|v) slab for one batch row

        var live = new List<(List<int> Seq, double LogP)> { ([Info.DecoderStartTokenId], 0) };
        var done = new List<(List<int> Seq, double LogP)>();
        var past = new DenseTensor<float>([layers, 2, 1, heads, 0, dk]);
        for (var t = 0; t < limit && live.Count > 0; t++)
        {
            var k = live.Count;
            var tok = new DenseTensor<long>([k, 1]);
            for (var i = 0; i < k; i++) tok[i, 0] = live[i].Seq[^1];
            var ck = new DenseTensor<float>([layers, 2, k, heads, s, dk]);
            var ckSpan = ck.Buffer.Span;
            for (var l = 0; l < layers * 2; l++)
                for (var i = 0; i < k; i++)
                    crossData.AsSpan(l * blockCross, blockCross).CopyTo(ckSpan.Slice((l * k + i) * blockCross, blockCross));
            var mask = new DenseTensor<long>(Enumerable.Repeat(1L, k * s).ToArray(), [k, s]);
            using var outs = _step!.Run([
                NamedOnnxValue.CreateFromTensor("token", tok),
                NamedOnnxValue.CreateFromTensor("position", new DenseTensor<long>(new[] { (long)t }, [1])),
                NamedOnnxValue.CreateFromTensor("attention_mask", mask),
                NamedOnnxValue.CreateFromTensor("cross_kv", ck),
                NamedOnnxValue.CreateFromTensor("past", past)]);
            var list = outs.ToList();
            var logits = list[0].AsTensor<float>();
            var present = list[1].AsTensor<float>();
            var vocab = logits.Dimensions[1];

            var candidates = new List<(List<int> Seq, double LogP, int Parent)>();
            for (var i = 0; i < k; i++)
            {
                var row = new float[vocab];
                for (var v = 0; v < vocab; v++) row[v] = logits[i, v];
                var max = row.Max();
                var logSum = max + Math.Log(row.Sum(x => Math.Exp(x - max)));
                foreach (var v in TopK(row, beams))
                    candidates.Add(([.. live[i].Seq, v], live[i].LogP + row[v] - logSum, i));
            }
            var next = new List<(List<int> Seq, double LogP, int Parent)>();
            foreach (var c in candidates.OrderByDescending(c => c.LogP))
            {
                if (c.Seq[^1] == Info.EosTokenId) done.Add((c.Seq, c.LogP));
                else next.Add(c);
                if (next.Count == beams) break;
            }
            live = next.Select(n => (n.Seq, n.LogP)).ToList();
            if (live.Count == 0) break;
            if (done.Count >= beams && done.Max(Norm) >= live.Select(Norm).Max()) break;

            // Reorder the cache rows to follow each surviving beam's parent.
            var tp = t + 1;
            var block = heads * tp * dk;
            var presentData = present.ToArray();
            past = new DenseTensor<float>([layers, 2, next.Count, heads, tp, dk]);
            var pSpan = past.Buffer.Span;
            for (var l = 0; l < layers * 2; l++)
                for (var i = 0; i < next.Count; i++)
                    presentData.AsSpan((l * k + next[i].Parent) * block, block).CopyTo(pSpan.Slice((l * next.Count + i) * block, block));
        }
        done.AddRange(live);
        return done.OrderByDescending(Norm).Take(beams).Select(x => (Decode(x.Seq.Skip(1)), Norm(x))).ToList();
    }

    static double Norm((List<int> Seq, double LogP) x) => x.LogP / Math.Max(1, x.Seq.Count - 1);

    static IEnumerable<int> TopK(float[] row, int k)
    {
        var idx = new List<int>(k + 1);
        foreach (var i in Enumerable.Range(0, row.Length))
        {
            if (idx.Count < k) { idx.Add(i); idx.Sort((a, b) => row[b].CompareTo(row[a])); continue; }
            if (row[i] <= row[idx[^1]]) continue;
            idx[^1] = i;
            idx.Sort((a, b) => row[b].CompareTo(row[a]));
        }
        return idx;
    }
}
