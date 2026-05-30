#!/usr/bin/env python3
"""Lightweight Speaker Diarization on Raspberry Pi 4 — Compression Pipeline.

Pyannote 3.1 -> structured pruning -> INT8 quantization -> DER eval (Colab)
+ latency/RAM (Pi, via bench_rpi.py).

Mental model — two machines, two jobs:

    Part A (this script)   Google Colab (free GPU)   compressed model + DER   -> accuracy columns
    Part B (bench_rpi.py)  real Raspberry Pi 4        latency + peak RAM       -> speed/memory columns

They connect through two files this script writes to Drive:
`segmentation_fp32.onnx` and `segmentation_int8.onnx`. You copy those to the Pi.

Cost: Colab free tier (a T4 session) is enough. Everything caches to Drive so
nothing is downloaded twice.

Honesty notes (read these):
  * Pyannote 3.1's actual segmentation model may not be the "WavLM front-end"
    your draft assumes (recent pyannote ships a SincNet/LSTM `segmentation-3.0`
    and a WeSpeaker embedder, not ECAPA). This script introspects the real
    layers instead of assuming, so it works either way — but update the paper's
    architecture claims to match what `step5_load_and_inspect` prints.
  * This is a faithful scaffold, not a one-click reproduction of specific
    numbers. Whatever DER/latency you measure here *is* your result.
  * DER is computed on the PyTorch models. The ONNX/INT8 file is for size + the
    Pi's latency; `step12_verify_fidelity` checks FP32-vs-INT8 output agreement.

Designed to run top-to-bottom in Google Colab. Run as `python rpi_diarization_compression.py`.
"""

import copy
import glob
import json
import os
import platform
import subprocess
import sys


# ---------------------------------------------------------------------------
# Step 0 — Confirm the runtime
# ---------------------------------------------------------------------------
# Make sure you actually got a GPU (Runtime -> Change runtime type -> T4).
# Pruning fine-tuning and DER eval are far faster on GPU; the rest works on CPU.
def step0_check_runtime():
    import torch

    print("Python :", platform.python_version())
    print("Torch  :", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU    :", torch.cuda.get_device_name(0))
    else:
        print("No GPU — fine for most steps, just slower. "
              "Runtime > Change runtime type > T4.")


# ---------------------------------------------------------------------------
# Step 1 — Mount Drive and point EVERY cache at it
# ---------------------------------------------------------------------------
# Colab wipes its local disk when the session ends. HuggingFace, Torch Hub, and
# the corpora all default to that ephemeral disk, so you'd re-download gigabytes
# every time. We redirect all of them to a single Drive folder. Set these env
# vars BEFORE importing any library that reads them.
ROOT = "/content/drive/MyDrive/diar_rpi"
PATHS = {
    "HF":      f"{ROOT}/cache/huggingface",   # gated model weights cache here ONCE
    "TORCH":   f"{ROOT}/cache/torch",
    "DATA":    f"{ROOT}/data",                # AMI / VoxConverse / CALLHOME
    "MODELS":  f"{ROOT}/models",              # pruned checkpoints
    "EXPORT":  f"{ROOT}/export",              # .onnx files -> copy to the Pi
    "RESULTS": f"{ROOT}/results",             # der.json etc.
}


def step1_mount_drive():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception:
        print("Not running in Colab — skipping Drive mount. "
              "Adjust ROOT above to a local path if needed.")

    for p in PATHS.values():
        os.makedirs(p, exist_ok=True)

    # Redirect caches so re-runs reuse Drive instead of re-downloading.
    os.environ["HF_HOME"] = PATHS["HF"]
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{PATHS['HF']}/hub"
    os.environ["TORCH_HOME"] = PATHS["TORCH"]
    os.environ["PYANNOTE_CACHE"] = f"{PATHS['HF']}/pyannote"
    print("Caches now live on Drive:")
    for k, v in PATHS.items():
        print(f"  {k:8s} -> {v}")


# ---------------------------------------------------------------------------
# Step 2 — Install pinned dependencies
# ---------------------------------------------------------------------------
# Why pin versions: diarization tooling breaks across releases. These match the
# ONNX Runtime 1.17.x line. `torch-pruning` does *real* structured channel
# removal (it tracks layer dependencies via a DepGraph, so removing an output
# channel automatically fixes the next layer's input channels — something
# `torch.nn.utils.prune` does NOT do; that only masks weights, no size/latency win).
def step2_install_deps():
    pkgs = [
        "pyannote.audio==3.1.1", "pyannote.metrics>=3.2",
        "torch-pruning>=1.3.0", "onnx>=1.15", "onnxruntime==1.17.1",
        "soundfile", "psutil",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
    print("Deps ready.")


# ---------------------------------------------------------------------------
# Step 3 — Authenticate with HuggingFace (gated models)
# ---------------------------------------------------------------------------
# `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` are gated.
# You must (1) accept the conditions on each model page in your browser once,
# and (2) provide a read token here. In Colab we pull it from Secrets
# (key icon -> HF_TOKEN) so it's never written in the file.
def step3_hf_login():
    from huggingface_hub import login

    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
    except Exception:
        import getpass
        hf_token = os.environ.get("HF_TOKEN") or getpass.getpass("HF token (hidden): ")

    login(hf_token)
    print("HF login OK. Accept terms at:")
    print("  https://hf.co/pyannote/speaker-diarization-3.1")
    print("  https://hf.co/pyannote/segmentation-3.0")
    return hf_token


# ---------------------------------------------------------------------------
# Step 4 — Datasets, with a download guard
# ---------------------------------------------------------------------------
# Each corpus is large. The helper downloads ONLY if the folder is empty, so
# re-running is instant after the first time.
#   * AMI         — free; headset-mix (ihm) audio + RTTM references.
#   * VoxConverse — RTTMs on GitHub; audio via the official script (large).
#   * CALLHOME    — LDC-licensed. Point CALLHOME_DIR at your existing copy; if
#                   absent, we skip it and run on AMI + VoxConverse.
# NOTE: replace the download URLs/commands with the exact ones from each
# corpus's site — they change, and CALLHOME must come from your licensed copy.
def _have_files(d, pattern="*"):
    return os.path.isdir(d) and len(
        glob.glob(os.path.join(d, "**", pattern), recursive=True)) > 0


def step4_prepare_datasets():
    data = PATHS["DATA"]
    dirs = {
        "AMI": f"{data}/ami",
        "VOX": f"{data}/voxconverse",
        "CALLHOME": f"{data}/callhome",  # <-- point to your LDC copy if you have it
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # ---- AMI (example; confirm current mirror) ----------------------------
    if not _have_files(dirs["AMI"], "*.rttm"):
        print("Downloading AMI references (run once)...")
        # RTTMs + UEMs for the diarization setup:
        subprocess.run(
            ["git", "clone", "-q",
             "https://github.com/pyannote/AMI-diarization-setup",
             f"{dirs['AMI']}/setup"],
            check=False)
        # Audio: use the official AMI download script for the ihm (headset-mix)
        # condition. See {AMI}/setup/pyannote/ for file lists. (Audio is many GB.)
    else:
        print("AMI already present — skipping download.")

    # ---- VoxConverse ------------------------------------------------------
    if not _have_files(dirs["VOX"], "*.rttm"):
        print("Downloading VoxConverse v0.3 references (run once)...")
        subprocess.run(
            ["git", "clone", "-q",
             "https://github.com/joonson/voxconverse",
             f"{dirs['VOX']}/refs"],
            check=False)
        # Audio via the repo's download instructions (YouTube-sourced; large).
    else:
        print("VoxConverse already present — skipping.")

    # ---- CALLHOME (licensed) ----------------------------------------------
    use_callhome = (_have_files(dirs["CALLHOME"], "*.rttm")
                    or _have_files(dirs["CALLHOME"], "*.sph"))
    print("CALLHOME available:", use_callhome,
          "(falls back to AMI+VoxConverse if False)")
    return dirs, use_callhome


# ---------------------------------------------------------------------------
# Build the evaluation file list
# ---------------------------------------------------------------------------
# All later steps iterate over a uniform list of (uri, audio_path, reference_rttm)
# triples, so adding/removing a corpus is trivial. Fill build_filelist with the
# actual paths once your audio is in place. Keep a small DEV_SUBSET for fast
# smoke-tests before committing to a full (slow) run.
def build_filelist(dirs, use_callhome=False):
    """Return [{'uri','audio','rttm'}...]. Adapt globs to your folder layout."""
    files = []
    # Example pattern — match RTTM stems to audio files:
    for rttm in glob.glob(f"{dirs['VOX']}/**/*.rttm", recursive=True):
        uri = os.path.splitext(os.path.basename(rttm))[0]
        wav = glob.glob(f"{dirs['VOX']}/**/{uri}.wav", recursive=True)
        if wav:
            files.append({"corpus": "VoxConverse", "uri": uri,
                          "audio": wav[0], "rttm": rttm})
    # ... repeat for AMI and (if use_callhome) CALLHOME ...
    return files


# ---------------------------------------------------------------------------
# Step 5 — Load Pyannote 3.1 and INSPECT its real architecture
# ---------------------------------------------------------------------------
# Why inspect instead of assume: drafts often talk about a "WavLM convolutional
# front-end" and "ECAPA-TDNN embeddings" — the shipped 3.1 pipeline may differ.
# We print the actual modules and collect the Conv1d layers (what structured
# pruning targets) so the code is correct regardless of internal architecture.
# Copy the printed layer names into your Methodology section (no more guessing).
def step5_load_and_inspect(hf_token):
    import torch
    import torch.nn as nn
    from pyannote.audio import Model, Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # The segmentation model is the heavy front-end we compress:
    seg_model = Model.from_pretrained(
        "pyannote/segmentation-3.0", use_auth_token=hf_token)
    seg_model.eval()

    conv_layers = [(n, m) for n, m in seg_model.named_modules()
                   if isinstance(m, nn.Conv1d)]
    total_params = sum(p.numel() for p in seg_model.parameters())
    print(f"Segmentation model: {total_params/1e6:.2f} M params, "
          f"{len(conv_layers)} Conv1d layers")
    for n, m in conv_layers:
        print(f"  {n:40s} in={m.in_channels:4d} out={m.out_channels:4d} "
              f"k={m.kernel_size}")
    # ^ Put these real numbers/names into the paper; fix front-end claims.
    return pipeline, seg_model, total_params


# ---------------------------------------------------------------------------
# Step 6 — A reusable DER evaluator (cached)
# ---------------------------------------------------------------------------
# A full DER sweep is slow. We save each result to results/der_<tag>.json;
# re-running returns the cached value instantly. DER uses a 0.25 s collar and
# ignores overlap — the NIST RT convention.
def eval_der(pipe, files, tag, force=False):
    from pyannote.database.util import load_rttm
    from pyannote.metrics.diarization import DiarizationErrorRate

    cache = f"{PATHS['RESULTS']}/der_{tag}.json"
    if os.path.exists(cache) and not force:
        r = json.load(open(cache))
        print(f"[cached] {tag}: {r['DER']*100:.2f}%")
        return r

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=True)
    per_file = {}
    for f in files:
        ref = load_rttm(f["rttm"])[f["uri"]]
        hyp = pipe(f["audio"])                       # run the pipeline
        per_file[f["uri"]] = metric(ref, hyp, uem=None)

    r = {"tag": tag, "DER": abs(metric), "n_files": len(files),
         "per_file": per_file}
    json.dump(r, open(cache, "w"), indent=2)
    print(f"{tag}: {r['DER']*100:.2f}% over {len(files)} files")
    return r


# ---------------------------------------------------------------------------
# Step 8 — Structured channel pruning (the real thing)
# ---------------------------------------------------------------------------
# Why torch-pruning + L1 importance: we rank each conv layer's output channels
# by L1 norm and remove the lowest `ratio` fraction. The DependencyGraph
# propagates each removal to every connected layer so the network stays valid
# and ACTUALLY gets smaller/faster (dense sub-network, no sparse masks). We
# prune at 0.30 and 0.59 to match the paper's two operating points.
#
# NOTE: if pruning the whole model drops params more than a "front-end only"
# claim, that's a contradiction a reviewer may flag — restrict ignored_layers
# to keep the embedder/clustering untouched, OR update the paper to say you
# prune model-wide.
def prune_model(base, ratio):
    import torch
    import torch.nn as nn
    import torch_pruning as tp

    model = copy.deepcopy(base).cpu().eval()
    example = torch.randn(1, 1, 16000 * 10)          # 10 s dummy waveform
    # Keep the final classification/head layer un-pruned (out_channels = #classes).
    ignored = [m for n, m in model.named_modules()
               if isinstance(m, nn.Conv1d) and m.out_channels <= 8]
    imp = tp.importance.MagnitudeImportance(p=1)      # L1 importance
    pruner = tp.pruner.MagnitudePruner(
        model, example, importance=imp,
        pruning_ratio=ratio, ignored_layers=ignored, global_pruning=False)
    pruner.step()                                     # perform the structured removal
    n = sum(p.numel() for p in model.parameters())
    print(f"ratio={ratio:.2f} -> {n/1e6:.2f} M params")
    torch.save(model, f"{PATHS['MODELS']}/seg_pruned_{int(ratio*100)}.pt")
    return model


# ---------------------------------------------------------------------------
# Step 9 — DER of the pruned models
# ---------------------------------------------------------------------------
# Swap the pruned segmentation model into a fresh pipeline and re-evaluate.
# (`_segmentation.model` is the hook pyannote exposes; if the attribute name
# differs in your version, the inspector in Step 5 shows the right path.)
def pipeline_with(seg, hf_token):
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    pipe._segmentation.model = seg.to(
        pipe.device if hasattr(pipe, "device") else "cpu")
    return pipe


# ---------------------------------------------------------------------------
# Step 10 — Export to ONNX
# ---------------------------------------------------------------------------
# ONNX is the portable format the Pi runs (via onnxruntime, no PyTorch needed),
# and it's what we quantize to INT8. opset 17 + dynamic axis on time lets the
# Pi feed arbitrary-length segments. We export both the FP32 baseline (for the
# Pi's ~1,840 ms row) and the 59%-pruned model.
def export_onnx(model, path):
    import torch

    model = model.cpu().eval()
    dummy = torch.randn(1, 1, 16000 * 10)
    torch.onnx.export(
        model, dummy, path, opset_version=17,
        input_names=["waveform"], output_names=["segmentation"],
        dynamic_axes={"waveform": {2: "samples"}, "segmentation": {1: "frames"}})
    print("exported", path, f"({os.path.getsize(path)/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Step 11 — INT8 static quantization (per-channel)
# ---------------------------------------------------------------------------
# Why static + per-channel: static quantization folds activation ranges in via
# a calibration pass, giving the size/latency win on the Pi's integer SIMD
# units; per-channel weights keep accuracy up across layers with different
# ranges. The CalibrationDataReader feeds ~200 real 10 s segments so the ranges
# reflect actual audio (synthetic noise mis-estimates ranges and hurts INT8 DER).
def _make_audio_calib_class():
    import numpy as np
    import soundfile as sf
    from onnxruntime.quantization import CalibrationDataReader

    class AudioCalib(CalibrationDataReader):
        """Yields real audio segments for INT8 calibration."""

        def __init__(self, files, n=200, sr=16000, secs=10):
            self.items, self.i = [], 0
            for f in files[:n]:
                wav, _ = sf.read(f["audio"])
                wav = wav[: sr * secs]
                if len(wav) < sr * secs:              # pad short clips
                    wav = np.pad(wav, (0, sr * secs - len(wav)))
                self.items.append(
                    {"waveform": wav.reshape(1, 1, -1).astype(np.float32)})

        def get_next(self):
            if self.i >= len(self.items):
                return None
            x = self.items[self.i]
            self.i += 1
            return x

    return AudioCalib


def step11_quantize_int8(pruned_path, files):
    from onnxruntime.quantization import QuantType, quantize_static

    audio_calib_cls = _make_audio_calib_class()
    int8_path = f"{PATHS['EXPORT']}/segmentation_int8.onnx"
    quantize_static(pruned_path, int8_path, audio_calib_cls(files),
                    per_channel=True, weight_type=QuantType.QInt8)
    print("INT8 size:",
          f"{os.path.getsize(int8_path)/1e6:.1f} MB  (this is your compressed model)")
    return int8_path


# ---------------------------------------------------------------------------
# Step 12 — Verify FP32-vs-INT8 fidelity
# ---------------------------------------------------------------------------
# Acceptance criterion: "DER agreement <= 0.1 pp between FP32 and INT8". We
# can't trivially run the full pipeline on ONNX, so we compare the segmentation
# logits directly on held-out files; a tiny mean-abs-error here is strong
# evidence the diarization output is unchanged. Report this as your
# quantization-fidelity check.
def _onnx_logits(path, wav):
    import onnxruntime as ort

    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s.run(None, {"waveform": wav})[0]


def step12_verify_fidelity(pruned_path, int8_path, dev_subset):
    import numpy as np
    import soundfile as sf

    errs = []
    for f in dev_subset:
        wav, _ = sf.read(f["audio"])
        wav = wav[:16000 * 10]
        wav = np.pad(wav, (0, 16000 * 10 - len(wav))).reshape(1, 1, -1).astype(np.float32)
        a, b = _onnx_logits(pruned_path, wav), _onnx_logits(int8_path, wav)
        errs.append(float(np.mean(np.abs(a - b))))
    print(f"FP32 vs INT8 mean-abs logit error: {np.mean(errs):.5f}  (smaller = better)")
    return float(np.mean(errs)) if errs else None


# ---------------------------------------------------------------------------
# Step 13 — (Optional) latency-aware fine-tuning
# ---------------------------------------------------------------------------
# Optional: recovers ~0.4 pp DER but costs GPU time. This is a sketch — wire in
# pyannote's training task + AMI loader for a real run. Three short epochs at
# lr=1e-5 on the pruned model is what the paper describes. Skip for a first pass.
def step13_finetune_optional(seg_59):
    # Pseudocode outline — replace with pyannote.audio's Segmentation task + AMI dataloader.
    # import torch
    # opt = torch.optim.AdamW(seg_59.parameters(), lr=1e-5)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3*len(loader))
    # for epoch in range(3):
    #     for batch in loader:
    #         loss = seg_59.training_step(batch)   # speaker-activity loss
    #         loss.backward(); opt.step(); sched.step(); opt.zero_grad()
    # torch.save(seg_59, f"{PATHS['MODELS']}/seg_59_finetuned.pt")
    print("Fine-tuning is optional; enable once the basic pipeline runs end-to-end.")


# ---------------------------------------------------------------------------
# Step 14 — Collect everything for the paper + the Pi
# ---------------------------------------------------------------------------
# One tidy summary on Drive, plus the two ONNX files you'll copy to the Pi.
def step14_collect_summary(total_params, fp32_path, pruned_path, int8_path):
    summary = {
        "segmentation_params_M": round(total_params / 1e6, 2),
        "fp32_onnx_MB": round(os.path.getsize(fp32_path) / 1e6, 1),
        "pruned59_onnx_MB": round(os.path.getsize(pruned_path) / 1e6, 1),
        "int8_onnx_MB": round(os.path.getsize(int8_path) / 1e6, 1),
        "der_results": "see results/der_*.json",
    }
    json.dump(summary, open(f"{PATHS['RESULTS']}/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("\nNow copy these to the Pi (from Drive):")
    print("  ", fp32_path)
    print("  ", int8_path)
    return summary


# ---------------------------------------------------------------------------
# Step 15 — Run the speed test ON THE PI
# ---------------------------------------------------------------------------
# Open bench_rpi.py ON the Raspberry Pi 4 (not here) after copying the two ONNX
# files:
#
#   pip3 install onnxruntime numpy psutil
#   python3 bench_rpi.py --model segmentation_fp32.onnx   # the ~1,840 ms baseline row
#   python3 bench_rpi.py --model segmentation_int8.onnx   # the compressed row
#
# It prints median latency, IQR, and peak RAM — your paper's speed/memory
# numbers. Copy rpi_bench_results.json back to Drive, then publish.
#
# Final reminder: report only the numbers *you* measure here and on the Pi. The
# cloud gives you a genuine model and genuine DER; the Pi gives you genuine
# latency/RAM. That's a defensible, reproducible result.


def main():
    step0_check_runtime()
    step1_mount_drive()
    step2_install_deps()
    hf_token = step3_hf_login()

    dirs, use_callhome = step4_prepare_datasets()
    files = build_filelist(dirs, use_callhome)
    dev_subset = files[:5]            # 5 files for a quick smoke test
    print(f"{len(files)} eval files found; using {len(dev_subset)} for smoke tests.")

    pipeline, seg_model, total_params = step5_load_and_inspect(hf_token)

    # Step 7 — Baseline DER. Run on dev_subset to smoke-test, then full FILES.
    eval_der(pipeline, dev_subset, tag="baseline_dev")    # quick check
    # eval_der(pipeline, files, tag="baseline_full")      # real number

    # Step 8 — prune at the two operating points.
    seg_30 = prune_model(seg_model, 0.30)
    seg_59 = prune_model(seg_model, 0.59)

    # Step 9 — DER of the pruned models.
    eval_der(pipeline_with(seg_30, hf_token), dev_subset, tag="pruned30_dev")
    eval_der(pipeline_with(seg_59, hf_token), dev_subset, tag="pruned59_dev")

    # Step 10 — export to ONNX.
    fp32_path = f"{PATHS['EXPORT']}/segmentation_fp32.onnx"
    pruned_path = f"{PATHS['EXPORT']}/segmentation_pruned59.onnx"
    export_onnx(seg_model, fp32_path)
    export_onnx(seg_59, pruned_path)

    # Step 11 — INT8 static quantization.
    int8_path = step11_quantize_int8(pruned_path, files or dev_subset)

    # Step 12 — fidelity check.
    step12_verify_fidelity(pruned_path, int8_path, dev_subset)

    # Step 13 — optional fine-tuning.
    step13_finetune_optional(seg_59)

    # Step 14 — collect summary.
    step14_collect_summary(total_params, fp32_path, pruned_path, int8_path)


if __name__ == "__main__":
    main()
