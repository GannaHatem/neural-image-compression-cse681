# M2 results review

Validated 216 neural records and 216 JPEG records across 24 full-resolution Kodak images.
Checks: PNG-based RGB PSNR, actual file bytes, bpp, and neural payload lengths. Errors: 0.
Independent NumPy decoding of kodim01, width 64, 8-bit weights agreed with the supplied PNG to within 1 intensity level; 0.000933% of RGB channel values differed due to numerical implementation differences. This is a spot check, not an independent decode of all 216 streams.
Total training time across 72 distinct image/width fits: 40.08 minutes. Three precisions reuse each fit; do not count that training three times.

| Width | Bits | Parameters | Mean bpp | Mean PSNR (dB) | Mean training (s) |
|---|---|---|---|---|---|
| 32 | 6 | 3267 | 0.0675 | 22.195 | 19.07 |
| 32 | 8 | 3267 | 0.0842 | 22.735 | 19.07 |
| 32 | 10 | 3267 | 0.1009 | 22.772 | 19.07 |
| 64 | 6 | 10627 | 0.1799 | 23.473 | 28.24 |
| 64 | 8 | 10627 | 0.2340 | 24.131 | 28.24 |
| 64 | 10 | 10627 | 0.2881 | 24.164 | 28.24 |
| 128 | 6 | 37635 | 0.5923 | 24.981 | 52.89 |
| 128 | 8 | 37635 | 0.7838 | 25.629 | 52.89 |
| 128 | 10 | 37635 | 0.9753 | 25.694 | 52.89 |

## Protocol and limits
- Arithmetic mean of per-image RGB PSNR, and arithmetic mean of bpp; all images have 393,216 pixels.
- Neural bytes include JSON metadata, bit-packed weights and header; no entropy coding.
- JPEG is 4:4:4. Curves are measured at fixed JPEG quality or fixed neural width/precision, not exact matched rates.
- jpeg_overlap_comparison.csv interpolates JPEG PSNR linearly in bpp per image, only inside measured support. It is an approximate matched-rate comparison; no extrapolation.
- Qualitative examples use the nearest sampled JPEG rate, with both actual rates shown.
- HEVC subsequently verified: see hevc_all.csv, hevc_summary.csv and the combined report.
- Per-image fitting intentionally optimizes the evaluated image; a conventional training/test split does not apply. Architecture/budget tuning should be disclosed.
- Actual format uses JSON endpoints and architecture metadata, deviating from the compact fixed-header estimate in M1.
- CPU timing is specific to this machine and software setup. JPEG timings cover in-memory encoding; neural encoding covers training plus serialization.

## Next steps
The HEVC baseline has now run and passed all 144 record checks. Add the M2 report using the course template, contribution statement and AI-assistance disclosure.