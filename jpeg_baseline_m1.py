import argparse
import csv
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(reference, reconstruction):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(reconstruction, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():
    parser = argparse.ArgumentParser(description='Milestone 1: Kodak JPEG baseline')
    parser.add_argument('--dataset', required=True, help='Directory containing kodim01.png ... kodim24.png')
    parser.add_argument('--output', default='jpeg_baseline_results', help='Separate output directory')
    parser.add_argument('--size', type=int, default=128, help='Square comparison resolution')
    parser.add_argument('--count', type=int, default=5, help='Number of Kodak images to process')
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    qualities = [10, 30, 50, 70, 90]
    for idx in range(1, args.count + 1):
        source = dataset / f'kodim{idx:02d}.png'
        if not source.is_file():
            raise FileNotFoundError(f'Missing Kodak image: {source}')
        with Image.open(source) as im:
            reference = im.convert('RGB').resize((args.size, args.size), Image.Resampling.LANCZOS)
        reference.save(output / f'kodim{idx:02d}_reference.png')
        for quality in qualities:
            buffer = io.BytesIO()
            start = time.perf_counter()
            reference.save(buffer, format='JPEG', quality=quality, subsampling=0)
            encode_seconds = time.perf_counter() - start
            encoded = buffer.getvalue()
            reconstructed = Image.open(io.BytesIO(encoded)).convert('RGB')
            reconstructed.load()
            (output / f'kodim{idx:02d}_q{quality}.jpg').write_bytes(encoded)
            value = psnr(reference, reconstructed)
            bpp = len(encoded) * 8 / (args.size * args.size)
            rows.append({'image': source.name, 'width': args.size, 'height': args.size,
                         'jpeg_quality': quality, 'psnr_db': round(value, 4),
                         'file_bytes': len(encoded), 'bpp': round(bpp, 5),
                         'encode_seconds': round(encode_seconds, 6)})
            print(f'{source.name} Q={quality:2d}  PSNR={value:6.2f} dB  bpp={bpp:6.3f}  encode={encode_seconds:.4f}s')
    csv_path = output / 'jpeg_baseline.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved {len(rows)} results to: {csv_path.resolve()}')
    print('Existing neural results were not modified.')


if __name__ == '__main__':
    main()
