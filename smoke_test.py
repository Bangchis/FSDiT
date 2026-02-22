"""
Smoke tests for FSDiT:
1) Dataset contract for precomputed supports.
2) DiT forward pass with and without sequence context.

Usage:
  python3 smoke_test.py --data_dir /path/to/miniimagenet_split --batch_size 2
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from dataset import build_dataset
from model import DiT

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")


def _resolve_train_dir(data_dir):
    train_dir = os.path.join(data_dir, "train")
    return train_dir if os.path.isdir(train_dir) else data_dir


def check_dataset_contract(data_dir, batch_size, image_size, num_sets):
    train_dir = _resolve_train_dir(data_dir)
    ds, _ = build_dataset(
        train_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_sets=num_sets,
        is_train=False,
        seed=0,
        debug_n=max(batch_size * 2, 2),
    )
    batch = next(iter(ds.as_numpy_iterator()))

    expected_keys = {"target", "supports_seq", "supports_pooled", "class_id"}
    got_keys = set(batch.keys())
    if got_keys != expected_keys:
        raise AssertionError(f"Dataset keys mismatch: got={got_keys}, expected={expected_keys}")

    target = batch["target"]
    supports_seq = batch["supports_seq"]
    supports_pooled = batch["supports_pooled"]

    if target.ndim != 4 or target.shape[-1] != 3:
        raise AssertionError(f"target must be (B,H,W,3), got {target.shape}")
    if supports_seq.shape[1:] != (5, 196, 768):
        raise AssertionError(f"supports_seq must be (B,5,196,768), got {supports_seq.shape}")
    if supports_pooled.shape[1:] != (5, 768):
        raise AssertionError(f"supports_pooled must be (B,5,768), got {supports_pooled.shape}")
    if not np.isfinite(target).all() or not np.isfinite(supports_seq).all() or not np.isfinite(supports_pooled).all():
        raise AssertionError("Non-finite values found in dataset batch.")

    print(
        "[OK] Dataset contract:"
        f" target={target.shape}, supports_seq={supports_seq.shape}, supports_pooled={supports_pooled.shape}"
    )
    return batch


def check_model_forward(batch):
    bsz = min(2, batch["target"].shape[0])
    x = jnp.asarray(batch["target"][:bsz], dtype=jnp.float32)
    y_seq = jnp.asarray(batch["supports_seq"][:bsz].reshape(bsz, -1, 768), dtype=jnp.float32)
    y_pooled = jnp.asarray(batch["supports_pooled"][:bsz].mean(axis=1), dtype=jnp.float32)
    t = jnp.linspace(0.1, 0.9, bsz, dtype=jnp.float32)

    dit = DiT(
        patch_size=8,
        hidden_size=64,
        depth=2,
        num_heads=2,
        mlp_ratio=1.0,
        siglip_dim=768,
        cond_dropout_prob=0.1,
    )

    rng = jax.random.PRNGKey(0)
    p_key, d_key = jax.random.split(rng)
    params = dit.init(
        {"params": p_key, "cond_dropout": d_key},
        x,
        t,
        y_pooled,
        y_seq=y_seq,
        train=True,
    )["params"]

    # New path: pooled + sequence context.
    out_ctx = dit.apply(
        {"params": params},
        x,
        t,
        y_pooled,
        y_seq=y_seq,
        train=False,
    )
    if out_ctx.shape != x.shape:
        raise AssertionError(f"DiT output shape mismatch (with context): {out_ctx.shape} vs {x.shape}")

    # Backward-compatible path: pooled only.
    out_pooled = dit.apply(
        {"params": params},
        x,
        t,
        y_pooled,
        train=False,
    )
    if out_pooled.shape != x.shape:
        raise AssertionError(f"DiT output shape mismatch (pooled-only): {out_pooled.shape} vs {x.shape}")

    if not np.isfinite(np.asarray(out_ctx)).all() or not np.isfinite(np.asarray(out_pooled)).all():
        raise AssertionError("Non-finite values in DiT forward outputs.")

    print(f"[OK] Model forward: with_context={out_ctx.shape}, pooled_only={out_pooled.shape}")


def main():
    parser = argparse.ArgumentParser(description="Run FSDiT smoke tests.")
    parser.add_argument("--data_dir", required=True, help="miniImageNet split root or train folder.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_sets", type=int, default=1)
    args = parser.parse_args()

    batch = check_dataset_contract(args.data_dir, args.batch_size, args.image_size, args.num_sets)
    check_model_forward(batch)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
