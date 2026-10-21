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
import shutil

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
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
    ids = tok(["parse: The dog slept."], return_tensors="pt")
    with torch.no_grad():
        hidden = Encoder(model)(ids.input_ids, ids.attention_mask)
        dec_ids = torch.tensor([[model.config.decoder_start_token_id, 0]])

        torch.onnx.export(Encoder(model), (ids.input_ids, ids.attention_mask), os.path.join(args.out, "encoder.onnx"),
                          input_names=["input_ids", "attention_mask"], output_names=["hidden"],
                          dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "hidden": {0: "b", 1: "s"}},
                          opset_version=17, dynamo=False)
        torch.onnx.export(Decoder(model), (dec_ids, hidden, ids.attention_mask), os.path.join(args.out, "decoder.onnx"),
                          input_names=["decoder_input_ids", "encoder_hidden", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"decoder_input_ids": {0: "b", 1: "t"}, "encoder_hidden": {0: "b", 1: "s"},
                                        "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b", 1: "t"}},
                          opset_version=17, dynamo=False)

    spm = getattr(tok, "vocab_file", None) or os.path.join(args.model, "spiece.model")
    shutil.copy(spm, os.path.join(args.out, "spiece.model"))
    files = {"encoder": "encoder.onnx", "decoder": "decoder.onnx"}
    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for part in ("encoder", "decoder"):
            quantize_dynamic(os.path.join(args.out, f"{part}.onnx"), os.path.join(args.out, f"{part}.int8.onnx"), weight_type=QuantType.QInt8)
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

    # Self-check: the ONNX decoder must reproduce PyTorch logits.
    import numpy as np
    import onnxruntime as ort
    enc = ort.InferenceSession(os.path.join(args.out, files["encoder"]))
    dec = ort.InferenceSession(os.path.join(args.out, files["decoder"]))
    h = enc.run(None, {"input_ids": ids.input_ids.numpy(), "attention_mask": ids.attention_mask.numpy()})[0]
    logits = dec.run(None, {"decoder_input_ids": dec_ids.numpy(), "encoder_hidden": h, "attention_mask": ids.attention_mask.numpy()})[0]
    with torch.no_grad():
        ref = Decoder(model)(dec_ids, hidden, ids.attention_mask).numpy()
    print("max |onnx - torch| logit difference:", float(np.abs(logits - ref).max()), "(int8 differs more; fine if argmax agrees)")
    print("argmax agrees:", bool((logits.argmax(-1) == ref.argmax(-1)).all()))
    print(f"exported to {args.out}: {meta}")


if __name__ == "__main__":
    main()
