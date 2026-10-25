# Project 3 — Neural Image Compression (Milestone 1)

CSE681: Advanced Digital Image and Video Processing

## Team
- Ganna Hatem Deghady — 2500014
- Gannah Mohamed Ghamery — 2501779
- Nour Ayman — 2501913

## What we tested

We tested the Fourier-feature MLP on Kodak image `kodim01.png`, resized to 128 × 128 pixels.

| Experiment | Training Iterations | Final PSNR | Training Time |
|---|---:|---:|---:|
| 3,000 iterations | 3,000 | 30.65 dB | 30.65 s |
| 5,000 iterations | 5,000 | 31.73 dB | 56.70 s |

The results of both experiments are saved in the `results_3000` and `results_5000` folders.

Increasing the training iterations improved the reconstruction quality by approximately 1.08 dB, but required more training time.

Both experiments used a network with 10,627 parameters. The estimated 8-bit weight bitrate is 5.19 bpp, which does not represent an actual compressed bitstream.

## Setup
```powershell
python -m pip install -r requirements.txt
```

FFmpeg with `libx265` is also needed for the HEVC baseline.

## Run the experiments
Replace the example paths with your local Kodak dataset path.

```powershell
python run_m1.py --image "G:\Postgrad\Advanced Digital Image and Video Processing\Data Set\kodim01.png" --steps 3000 --out results_new
python jpeg_baseline_m1.py --dataset "G:\Postgrad\Advanced Digital Image and Video Processing\Data Set" --output jpeg_baseline_results_new
python hevc_baseline_m1.py --dataset "G:\Postgrad\Advanced Digital Image and Video Processing\Data Set" --output hevc_baseline_results_new
```

Use new output directories so that reruns do not overwrite the original experimental evidence.

## Files
- `run_m1.py`: neural reconstruction and training history
- `jpeg_baseline_m1.py`: five-image JPEG benchmark
- `hevc_baseline_m1.py`: five-image HEVC intra-only benchmark
- `jpeg_baseline.csv`, `hevc_baseline.csv`: previously measured results
- `summary.json`, `neural_reconstruction.png`: neural experiment evidence
- `M1_Proposal.pdf`, `M1_Proposal.tex`, `references.bib`: proposal and sources

## Measurement notes
JPEG and HEVC baselines are evaluated on resized 128×128 images. The neural 8-bit weight bpp is a theoretical estimate; no quantized bitstream or decoder has been implemented yet. The JPEG measurements in `run_m1.py` are at the original image resolution and should not be used for direct comparison with the 128×128 benchmarks.
