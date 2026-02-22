"""
Precompute SigLIP2 embeddings for miniImageNet support images.

Output format (saved next to each image):
  <image_name>.npz with keys:
    - seq:    (196, 768)
    - pooled: (768,)
"""

import argparse
import os
import time

import jax
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from encoder import SigLIP2Encoder

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")


def _is_image_file(name):
    return name.lower().endswith((".jpg", ".jpeg", ".png"))


def _collect_images(data_dir):
    paths = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if _is_image_file(fname):
                paths.append(os.path.join(root, fname))
    paths.sort()
    return paths


def _read_image(path, image_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img.numpy()[None, ...]


def _validate_outputs(seq_embs, pooled_emb):
    if seq_embs.shape != (196, 768):
        raise ValueError(f"Expected seq shape (196, 768), got {seq_embs.shape}")
    if pooled_emb.shape != (768,):
        raise ValueError(f"Expected pooled shape (768,), got {pooled_emb.shape}")
    if not np.isfinite(seq_embs).all() or not np.isfinite(pooled_emb).all():
        raise ValueError("NaN/Inf detected in embeddings")


def main():
    parser = argparse.ArgumentParser(description="Precompute SigLIP2 sequence+pooled embeddings.")
    parser.add_argument("--data_dir", required=True, help="Root folder containing class folders.")
    parser.add_argument("--image_size", type=int, default=224, help="SigLIP resolution.")
    parser.add_argument("--variant", default="B/16", help="SigLIP2 vision variant.")
    parser.add_argument("--ckpt_path", default=None, help="Optional local SigLIP checkpoint .npz.")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "float32"),
        help="Saved embedding dtype.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing .npz files.")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue when a file fails. Default is fail-fast.",
    )
    args = parser.parse_args()

    t0 = time.time()
    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    image_paths = _collect_images(args.data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {args.data_dir}")

    print(f"Found {len(image_paths)} images under {args.data_dir}")
    print("Loading SigLIP2 encoder...")
    siglip = SigLIP2Encoder.create(
        ckpt_path=args.ckpt_path,
        variant=args.variant,
        res=args.image_size,
    )

    done = 0
    skipped = 0
    failed = 0
    for img_path in tqdm(image_paths, dynamic_ncols=True):
        npz_path = os.path.splitext(img_path)[0] + ".npz"
        if os.path.exists(npz_path) and not args.overwrite:
            skipped += 1
            continue

        try:
            img = _read_image(img_path, args.image_size)
            seq_embs, pooled_emb = siglip._encode_both(img)
            seq_embs = np.asarray(jax.device_get(seq_embs)).squeeze(0)
            pooled_emb = np.asarray(jax.device_get(pooled_emb)).squeeze(0)
            _validate_outputs(seq_embs, pooled_emb)
            np.savez(
                npz_path,
                seq=seq_embs.astype(np_dtype),
                pooled=pooled_emb.astype(np_dtype),
            )
            done += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {img_path}: {exc}")
            if not args.continue_on_error:
                raise

    dt = time.time() - t0
    print(
        f"Done in {dt:.1f}s | saved={done}, skipped={skipped}, failed={failed}, "
        f"dtype={args.dtype}"
    )


if __name__ == "__main__":
    main()
