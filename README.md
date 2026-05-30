# Lightweight Speaker Diarization on Raspberry Pi 4

Compression of Pyannote 3.1 (structured pruning + INT8 quantization) benchmarked on a
Raspberry Pi 4. **Raspberry Pi is the only target hardware** in this experiment.

## The one idea to keep in your head

Two machines, two jobs. They are **not interchangeable**:

| Part | Runs on | Measures | Why it must be there |
|------|---------|----------|----------------------|
| A — `RPi_Diarization_Compression.ipynb` | **Google Colab** (free T4) | model + **DER** | accuracy is hardware-independent → cloud is fine & free |
| B — `bench_rpi.py` | **your Raspberry Pi 4** | **latency + peak RAM** | these only exist on the Pi's ARM Cortex-A72 CPU |

You **cannot** measure Pi latency in Colab (it's x86/GPU). The two parts meet through two
files Part A writes to Drive — `segmentation_fp32.onnx`, `segmentation_int8.onnx` — which
you copy to the Pi for Part B.

## Cost (minimum)

- **Colab**: free tier is enough. (Pro only if full-corpus DER runs time out.)
- **Drive**: ~15 GB free; corpora may need a paid tier or external storage.
- **Hardware**: the Pi 4 you already own. No other spend.

## Run order

1. **Revoke the GitHub token you pasted earlier** and make a new one. Add it to Colab
   Secrets as `GH_TOKEN`. Add your HuggingFace read token as `HF_TOKEN`.
2. Accept model terms (browser, once): `pyannote/speaker-diarization-3.1`,
   `pyannote/segmentation-3.0`.
3. Open the notebook in Colab → run Steps 0–14. First run downloads weights + data to
   Drive; **every later run reuses the cache (nothing downloads twice)**.
4. Copy `segmentation_fp32.onnx` and `segmentation_int8.onnx` from
   `MyDrive/diar_rpi/export/` to the Pi.
5. On the Pi: `pip3 install onnxruntime numpy psutil`, then run `bench_rpi.py` for each
   model. Copy `rpi_bench_results.json` back to Drive.
6. `upload_to_github.py` publishes code + results (uses the secret token, never a literal).

## How "don't download twice" works

The notebook's Step 1 redirects `HF_HOME`, `TORCH_HOME`, `PYANNOTE_CACHE`, and all dataset
folders to `MyDrive/diar_rpi/`. Downloads are also guarded (`if folder empty: download`),
and DER results are cached to JSON. A fresh Colab session re-mounts Drive and finds
everything already there.

## Lightning.ai instead of Colab?

- Replace `from google.colab import drive` / `userdata` with Lightning's persistent
  Studio storage (`/teamspace/studios/this_studio`) and `getpass` for tokens.
- Point `ROOT` at the persistent path instead of `/content/drive/MyDrive`.
- Everything else (pruning, ONNX, quantization, DER, the Pi script) is identical.

## Honesty / paper integrity

- Step 5 prints the **real** architecture — update the draft's "WavLM front-end / ECAPA /
  27.4 M params" claims to match what it shows.
- Report only numbers you actually measure here and on the Pi. This scaffold gives a
  genuine, reproducible result; that is what survives peer review.

## Files

- `RPi_Diarization_Compression.ipynb` — Part A (Colab).
- `bench_rpi.py` — Part B (Raspberry Pi).
- `upload_to_github.py` — safe publish (no hardcoded token).
