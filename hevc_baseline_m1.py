"""Measure HEVC intra-only compression on five Kodak images.

Requires FFmpeg built with the libx265 encoder.
"""

import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image


def run_ffmpeg(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {' '.join(map(str, command))}\n{result.stderr[-3000:]}")


def calculate_psnr(reference, decoded):
    error = reference.astype(np.float64) - decoded.astype(np.float64)
    mse = np.mean(error ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def encode_image(ffmpeg, reference_path, bitstream_path, crf):
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(reference_path), "-frames:v", "1",
        "-pix_fmt", "yuv444p", "-c:v", "libx265", "-preset", "medium",
        "-x265-params", "keyint=1:min-keyint=1:scenecut=0:log-level=error",
        "-crf", str(crf), "-f", "hevc", str(bitstream_path),
    ]
    start = time.perf_counter()
    run_ffmpeg(command)
    return time.perf_counter() - start


def decode_image(ffmpeg, bitstream_path, output_path):
    run_ffmpeg([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(bitstream_path), "-frames:v", "1", str(output_path),
    ])
    with Image.open(output_path) as image:
        return np.asarray(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description="Kodak HEVC intra baseline")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("hevc_baseline_results"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--crf", nargs="+", type=int, default=[10, 20, 30, 40, 45])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    measurements = []

    for number in range(1, 6):
        name = f"kodim{number:02d}"
        source_path = args.dataset / f"{name}.png"
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        with Image.open(source_path) as source:
            reference = source.convert("RGB").resize(
                (args.size, args.size), Image.Resampling.LANCZOS
            )
        reference_path = args.output / f"{name}_reference.png"
        reference.save(reference_path)
        reference_pixels = np.asarray(reference)

        for crf in args.crf:
            bitstream_path = args.output / f"{name}_crf{crf}.hevc"
            decoded_path = args.output / f"{name}_crf{crf}_decoded.png"

            seconds = encode_image(args.ffmpeg, reference_path, bitstream_path, crf)
            decoded_pixels = decode_image(args.ffmpeg, bitstream_path, decoded_path)
            if decoded_pixels.shape != reference_pixels.shape:
                raise ValueError("Decoded image resolution does not match the reference")

            file_bytes = bitstream_path.stat().st_size
            row = {
                "image": name,
                "crf": crf,
                "width": args.size,
                "height": args.size,
                "psnr_db": round(calculate_psnr(reference_pixels, decoded_pixels), 4),
                "bpp": round(8 * file_bytes / (args.size ** 2), 5),
                "bytes": file_bytes,
                "encoding_seconds": round(seconds, 4),
                "pixel_format": "yuv444p",
                "codec": "libx265",
                "preset": "medium",
            }
            measurements.append(row)
            print(f"{name} CRF {crf}: PSNR={row['psnr_db']:.2f} dB, "
                  f"bpp={row['bpp']:.3f}, encode={seconds:.2f}s")

    csv_path = args.output / "hevc_baseline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=measurements[0].keys())
        writer.writeheader()
        writer.writerows(measurements)

    print(f"Saved {len(measurements)} measurements to {csv_path}")
    print("Note: raw HEVC bitstream headers count toward bpp.")


if __name__ == "__main__":
    main()
