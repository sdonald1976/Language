"""Export a fine-tuned T5-family checkpoint to ONNX for the C# engine (Microsoft.ML.OnnxRuntime).

    python training/export_onnx.py --model runs/c1-base/best --out models/c1-base [--int8]

Produces:
    encoder.onnx   input_ids, attention_mask                         -> hidden
    decoder.onnx   decoder_input_ids, encoder_hidden, attention_mask -> logits   (no KV cache; simple and robust)
    spiece.model   SentencePiece tokenizer, loaded by Microsoft.ML.Tokenizers
    engine.json    special token ids and limits
With --int8 it also writes encoder.int8.onnx / decoder.int8.onnx (dynamic quantization) and the engine uses them.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.get_encoder()

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, decoder_input_ids, encoder_hidden, attention_mask):
        out = self.model(encoder_outputs=(encoder_hidden,), attention_mask=attention_mask,
                         decoder_input_ids=decoder_input_ids, use_cache=False)
        return out.logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    # Eager attention: the SDPA path has Python branches on sequence lengths that tracing would freeze.
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, attn_implementation="eager").eval()
    ids = tok(["parse: The tired dog that we adopted last spring slept under the old kitchen table again."], return_tensors="pt")
    with torch.no_grad():
        hidden = Encoder(model)(ids.input_ids, ids.attention_mask)
        dec_ids = torch.tensor([[model.config.decoder_start_token_id] + list(range(100, 107))])

        torch.onnx.export(Encoder(model), (ids.input_ids, ids.attention_mask), os.path.join(args.out, "encoder.onnx"),
                          input_names=["input_ids", "attention_mask"], output_names=["hidden"],
                          dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "hidden": {0: "b", 1: "s"}},
                          opset_version=17, dynamo=False)
        torch.onnx.export(Decoder(model), (dec_ids, hidden, ids.attention_mask), os.path.join(args.out, "decoder.onnx"),
                          input_names=["decoder_input_ids", "encoder_hidden", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"decoder_input_ids": {0: "b", 1: "t"}, "encoder_hidden": {0: "b", 1: "s"},
                                        "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b", 1: "t"}},
                          opset_version=17, dynamo=False)

    # Microsoft.ML.Tokenizers cannot load a SentencePiece model whose bos_id is -1 (T5 has no BOS token).
    # The engine never adds BOS, so pointing bos_id at <pad> (0) changes no token ids; parity is checked in RUNNING.md.
    from sentencepiece import sentencepiece_model_pb2 as spm_pb
    spm = getattr(tok, "vocab_file", None) or os.path.join(args.model, "spiece.model")
    proto = spm_pb.ModelProto()
    with open(spm, "rb") as f:
        proto.ParseFromString(f.read())
    if proto.trainer_spec.bos_id < 0:
        proto.trainer_spec.bos_id = 0
    with open(os.path.join(args.out, "spiece.model"), "wb") as f:
        f.write(proto.SerializeToString())
    files = {"encoder": "encoder.onnx", "decoder": "decoder.onnx"}
    if args.int8:
        # T5's feed-forward output projections (wo) and the LM head have outlier activations that int8 ruins;
        # quantize every other MatMul per channel (≈2.5x smaller, argmax-equivalent in our checks).
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for part in ("encoder", "decoder"):
            src = os.path.join(args.out, f"{part}.onnx")
            graph = onnx.load(src, load_external_data=False).graph
            exclude = [n.name for n in graph.node if n.op_type == "MatMul" and ("/wo/" in n.name or "lm_head" in n.name)]
            quantize_dynamic(src, os.path.join(args.out, f"{part}.int8.onnx"), weight_type=QuantType.QInt8,
                             op_types_to_quantize=["MatMul"], per_channel=True, nodes_to_exclude=exclude)
            files[part] = f"{part}.int8.onnx"
    meta = {
        "decoder_start_token_id": model.config.decoder_start_token_id,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "vocab_size": model.config.vocab_size,
        "max_source": 384,
        "max_target": 384,
        "files": files,
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    }
    with open(os.path.join(args.out, "engine.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Self-check on a real greedy decode: fp32 must match PyTorch; int8 must pick the same tokens.
    import numpy as np
    import onnxruntime as ort
    probe = tok(["parse: If the snuggletron doesn't charge, Maria won't hug it."], return_tensors="pt")
    reference = AutoModelForSeq2SeqLM.from_pretrained(args.model, attn_implementation="eager").eval()  # tracing leaves state behind
    with torch.no_grad():
        seq = reference.generate(**probe, max_new_tokens=60)[:, :-1]
        ref = reference(input_ids=probe.input_ids, attention_mask=probe.attention_mask, decoder_input_ids=seq).logits.numpy()
    for label, enc_file, dec_file in [("fp32", "encoder.onnx", "decoder.onnx"), ("int8", files["encoder"], files["decoder"])]:
        if label == "int8" and not args.int8:
            continue
        enc = ort.InferenceSession(os.path.join(args.out, enc_file))
        dec = ort.InferenceSession(os.path.join(args.out, dec_file))
        h = enc.run(None, {"input_ids": probe.input_ids.numpy(), "attention_mask": probe.attention_mask.numpy()})[0]
        logits = dec.run(None, {"decoder_input_ids": seq.numpy(), "encoder_hidden": h, "attention_mask": probe.attention_mask.numpy()})[0]
        agree = float((logits.argmax(-1) == ref.argmax(-1)).mean())
        print(f"{label}: max |onnx - torch| = {float(np.abs(logits - ref).max()):.4f}, argmax agreement = {agree:.3f}")
        if label == "int8" and agree < 0.98:
            print("int8 disagrees with the PyTorch model; engine.json will use the fp32 files instead")
            meta["files"] = {"encoder": "encoder.onnx", "decoder": "decoder.onnx"}
            with open(os.path.join(args.out, "engine.json"), "w") as f:
                json.dump(meta, f, indent=2)
    print(f"exported to {args.out}: {meta}")


if __name__ == "__main__":
    main()
