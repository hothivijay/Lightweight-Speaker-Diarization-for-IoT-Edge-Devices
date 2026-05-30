#!/usr/bin/env python3
# =============================================================================
# bench_rpi.py  —  RUNS ON THE RASPBERRY PI 4 (not in Colab)
# -----------------------------------------------------------------------------
# WHY THIS FILE EXISTS:
#   Latency and peak-RAM are the ONLY numbers in your paper that are
#   hardware-specific. They cannot be measured in Colab (x86/GPU). This script
#   loads the ONNX models you exported in the cloud and times them on the Pi's
#   ARM Cortex-A72 CPU exactly as the paper describes: median wall-clock time
#   over 100 passes on a 10-second, 16 kHz mono segment, first 10 discarded as
#   warm-up; peak RSS sampled in a background thread.
#
# WHAT YOU COPY TO THE PI (from Google Drive):
#   segmentation_fp32.onnx   (the full-precision baseline, for the 1,840 ms row)
#   segmentation_int8.onnx   (pruned + INT8, the compressed model)
#
# SETUP ON THE PI (Raspberry Pi OS 64-bit, do this once):
#   sudo apt update && sudo apt install -y python3-pip
#   pip3 install onnxruntime numpy psutil          # CPU-only build; no torch needed
#   # (onnxruntime ships ARM64 wheels; install the 64-bit OS so the wheel exists)
#
# RUN:
#   python3 bench_rpi.py --model segmentation_int8.onnx --runs 100 --warmup 10
#   python3 bench_rpi.py --model segmentation_fp32.onnx --runs 100 --warmup 10
# =============================================================================

import argparse, json, time, threading, statistics, os
import numpy as np
import onnxruntime as ort
import psutil


def peak_rss_sampler(stop_flag, samples, interval=0.01):
    """Sample this process's resident memory every `interval` s in a thread.
    WHY a thread: peak RAM happens *during* inference; polling after the fact
    would miss the transient allocation peak. 10 ms matches the paper's method."""
    proc = psutil.Process(os.getpid())
    while not stop_flag["stop"]:
        samples.append(proc.memory_info().rss)
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the .onnx model")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=10.0, help="segment length")
    ap.add_argument("--sr", type=int, default=16000, help="sample rate")
    ap.add_argument("--out", default="rpi_bench_results.json")
    args = ap.parse_args()

    # --- Force single-threaded, deterministic CPU execution -------------------
    # WHY: the Pi has 4 cores; leaving ORT to auto-thread makes latency noisy and
    # non-reproducible. Pin threads so every run is comparable. If you WANT to
    # report multi-core latency, set intra_op_num_threads=4 and say so in the paper.
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4          # use all 4 A72 cores; document this choice
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(args.model, sess_options=so,
                                providers=["CPUExecutionProvider"])

    # --- Build one fixed 10 s dummy input -------------------------------------
    # WHY a fixed synthetic segment: latency depends on TENSOR SHAPE, not on the
    # audio content. A fixed (1, 1, sr*seconds) tensor gives a clean, repeatable
    # measurement and removes file-I/O from the timing loop. Shape must match the
    # ONNX model's expected input — adjust the name/shape if export differed.
    inp_name = sess.get_inputs()[0].name
    n_samples = int(args.sr * args.seconds)
    x = np.random.randn(1, 1, n_samples).astype(np.float32)

    # --- Peak RAM sampling thread ---------------------------------------------
    stop_flag = {"stop": False}
    rss_samples = []
    t = threading.Thread(target=peak_rss_sampler, args=(stop_flag, rss_samples))
    t.start()

    # --- Warm-up (discarded): first runs include JIT / cache effects ----------
    for _ in range(args.warmup):
        sess.run(None, {inp_name: x})

    # --- Timed runs -----------------------------------------------------------
    times_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: x})
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    stop_flag["stop"] = True
    t.join()

    # --- Report median + IQR (the paper's convention), not mean --------------
    # WHY median/IQR: latency distributions are right-skewed (OS scheduling
    # spikes). Median + IQR is robust; mean ± std would be inflated by outliers.
    q1, med, q3 = np.percentile(times_ms, [25, 50, 75])
    peak_ram_mb = max(rss_samples) / (1024 * 1024) if rss_samples else float("nan")

    result = {
        "model": os.path.basename(args.model),
        "runs": args.runs,
        "latency_ms_median": round(float(med), 1),
        "latency_ms_iqr": round(float(q3 - q1), 1),
        "latency_ms_min": round(float(min(times_ms)), 1),
        "latency_ms_max": round(float(max(times_ms)), 1),
        "peak_ram_mb": round(peak_ram_mb, 1),
        "threads": so.intra_op_num_threads,
    }
    print(json.dumps(result, indent=2))

    # Append so you can run fp32 then int8 and keep both rows in one file.
    history = []
    if os.path.exists(args.out):
        history = json.load(open(args.out))
    history.append(result)
    json.dump(history, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}  (copy this back to Drive for the paper tables)")


if __name__ == "__main__":
    main()
