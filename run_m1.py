"""Train a small coordinate-based network to reconstruct one Kodak image.

This is a feasibility experiment, not a finished neural compression codec.
"""

import argparse
import csv
import io
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn


class FourierMLP(nn.Module):
    """Predict RGB values from normalized pixel coordinates."""

    def __init__(self, width=64, frequencies=8):
        super().__init__()
        frequency_values = (2.0 ** torch.arange(frequencies)) * math.pi
        self.register_buffer("freqs", frequency_values)

        input_size = 4 * frequencies
        self.net = nn.Sequential(
            nn.Linear(input_size, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 3),
            nn.Sigmoid(),
        )

    def forward(self, coordinates):
        angles = coordinates[..., None] * self.freqs
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        features = features.flatten(start_dim=1)
        return self.net(features)


def calculate_psnr(reference, reconstructed):
    reference = reference.astype(np.float64) / 255.0
    reconstructed = reconstructed.astype(np.float64) / 255.0
    mse = np.mean((reference - reconstructed) ** 2)
    return float("inf") if mse == 0 else -10 * math.log10(mse)


def save_original_size_jpeg_baseline(image, output_folder):
    """Convenience baseline; not directly comparable to resized neural output."""
    original = np.asarray(image)
    measurements = []

    for quality in [10, 30, 50, 70, 90]:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=0)
        encoded = buffer.getvalue()
        decoded = Image.open(io.BytesIO(encoded)).convert("RGB")

        measurements.append({
            "quality": quality,
            "bytes": len(encoded),
            "bpp": 8 * len(encoded) / (image.width * image.height),
            "psnr_db": calculate_psnr(original, np.asarray(decoded)),
        })

    with (output_folder / "jpeg_baseline.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=measurements[0].keys())
        writer.writeheader()
        writer.writerows(measurements)


def prepare_pixels(image, size, device):
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    pixels = torch.from_numpy(np.asarray(resized).copy()).float()
    pixels = pixels.reshape(-1, 3).to(device) / 255.0

    positions = torch.linspace(0, 1, size)
    y_grid, x_grid = torch.meshgrid(positions, positions, indexing="ij")
    coordinates = torch.stack([x_grid, y_grid], dim=-1)
    coordinates = coordinates.reshape(-1, 2).to(device)
    return resized, coordinates, pixels


def reconstruct_all_pixels(model, coordinates):
    """Evaluate in chunks to keep memory use reasonable."""
    with torch.no_grad():
        return torch.cat([model(chunk) for chunk in coordinates.split(4096)])


def train_model(model, coordinates, pixels, steps, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        sample_count = min(4096, len(coordinates))
        indices = torch.randint(0, len(coordinates), (sample_count,), device=device)
        prediction = model(coordinates[indices])
        loss = ((prediction - pixels[indices]) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == steps:
            full_prediction = reconstruct_all_pixels(model, coordinates)
            mse = ((full_prediction - pixels) ** 2).mean().item()
            psnr = -10 * math.log10(max(mse, 1e-12))
            seconds = time.perf_counter() - start_time
            history.append({"step": step, "psnr_db": psnr, "seconds": seconds})
            print(f"Step {step}: PSNR={psnr:.2f} dB, time={seconds:.1f}s")

    return history, time.perf_counter() - start_time


def save_results(output_folder, image, model, coordinates, history, elapsed, size, steps, device):
    predicted = reconstruct_all_pixels(model, coordinates)
    predicted = predicted.reshape(size, size, 3).cpu().numpy()
    reconstructed = np.uint8(np.clip(predicted * 255, 0, 255).round())

    Image.fromarray(reconstructed).save(output_folder / "neural_reconstruction.png")
    image.save(output_folder / "resized_target.png")
    checkpoint = output_folder / "model_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "device": device,
        "parameters": parameter_count,
        "estimated_8bit_weight_bpp": parameter_count * 8 / (size * size),
        "checkpoint_bytes_not_compressed_bitstream": checkpoint.stat().st_size,
        "image_size": size,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "note": "Neural bpp is an estimate only, not an actual compressed codec. No decoder/quantization implemented.",
    }
    (output_folder / "summary.json").write_text(json.dumps(summary, indent=2))

    with (output_folder / "training_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    plt.figure()
    plt.plot([row["seconds"] for row in history],
             [row["psnr_db"] for row in history], marker="o")
    plt.xlabel("Training time (s)")
    plt.ylabel("PSNR (dB)")
    plt.tight_layout()
    plt.savefig(output_folder / "psnr_vs_time.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Single-image neural reconstruction")
    parser.add_argument("--image", required=True, help="Path to a Kodak PNG image")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    output_folder = Path(args.out)
    output_folder.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    save_original_size_jpeg_baseline(image, output_folder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resized_image, coordinates, pixels = prepare_pixels(image, args.size, device)
    model = FourierMLP(width=args.width).to(device)

    history, elapsed = train_model(model, coordinates, pixels, args.steps, device)
    save_results(output_folder, resized_image, model, coordinates,
                 history, elapsed, args.size, args.steps, device)
    print("Results saved to", output_folder)


if __name__ == "__main__":
    main()
