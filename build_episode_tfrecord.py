"""
Build episode-level TFRecord shards from miniImageNet + precomputed support embeddings.

Each record stores:
  - target_path (string)
  - class_id (int)
  - supports_pooled (bytes; float16 [5,768])
  - supports_seq (bytes; float16 [5,196,768]) if --store_seq=1 else empty bytes
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset import build_episode_table, _interleave_by_class


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _npz_path_for(image_path, split_dir, embedding_split_dir):
    rel = os.path.relpath(image_path, split_dir)
    return os.path.join(embedding_split_dir, os.path.splitext(rel)[0] + ".npz")


def _load_support_arrays(support_paths, split_dir, embedding_split_dir, store_seq):
    pooled_list = []
    seq_list = []
    for p in support_paths:
        npz_path = _npz_path_for(p, split_dir, embedding_split_dir)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing embedding npz: {npz_path}")
        d = np.load(npz_path)
        pooled = d["pooled"].astype(np.float16, copy=False)
        if pooled.shape != (768,):
            raise ValueError(f"Invalid pooled shape in {npz_path}: {pooled.shape}")
        pooled_list.append(pooled)
        if store_seq:
            seq = d["seq"].astype(np.float16, copy=False)
            if seq.shape != (196, 768):
                raise ValueError(f"Invalid seq shape in {npz_path}: {seq.shape}")
            seq_list.append(seq)

    pooled_arr = np.stack(pooled_list, axis=0)  # (5, 768)
    if store_seq:
        seq_arr = np.stack(seq_list, axis=0)     # (5, 196, 768)
    else:
        seq_arr = None
    return pooled_arr, seq_arr


def build_split(split, data_dir, embeddings_dir, out_dir, num_sets, seed, num_shards, store_seq, compression):
    split_dir = os.path.join(data_dir, split)
    embedding_split_dir = os.path.join(embeddings_dir, split)
    if not os.path.isdir(split_dir):
        print(f"[Skip] split '{split}' not found at {split_dir}")
        return
    if not os.path.isdir(embedding_split_dir):
        raise FileNotFoundError(f"Missing embedding split dir: {embedding_split_dir}")

    episodes, class_names = build_episode_table(split_dir, num_sets=num_sets, seed=seed)
    episodes = _interleave_by_class(episodes, len(class_names), seed + 1)
    print(f"[{split}] classes={len(class_names)} episodes={len(episodes)}")

    split_out = os.path.join(out_dir, split)
    os.makedirs(split_out, exist_ok=True)

    writers = []
    shard_paths = []
    tf_opts = tf.io.TFRecordOptions(compression_type=compression) if compression else None
    for i in range(num_shards):
        p = os.path.join(split_out, f"{split}-{i:05d}-of-{num_shards:05d}.tfrecord")
        shard_paths.append(p)
        if tf_opts:
            writers.append(tf.io.TFRecordWriter(p, options=tf_opts))
        else:
            writers.append(tf.io.TFRecordWriter(p))

    try:
        for idx, (target_path, support_paths, class_id) in enumerate(tqdm(episodes, desc=f"write-{split}", dynamic_ncols=True)):
            pooled_arr, seq_arr = _load_support_arrays(
                support_paths, split_dir, embedding_split_dir, store_seq=store_seq
            )
            seq_bytes = seq_arr.tobytes(order="C") if store_seq else b""

            ex = tf.train.Example(features=tf.train.Features(feature={
                "target_path": _bytes_feature(target_path.encode("utf-8")),
                "class_id": _int64_feature(class_id),
                "supports_pooled": _bytes_feature(pooled_arr.tobytes(order="C")),
                "supports_seq": _bytes_feature(seq_bytes),
            }))
            writers[idx % num_shards].write(ex.SerializeToString())
    finally:
        for w in writers:
            w.close()

    meta = {
        "split": split,
        "num_classes": len(class_names),
        "num_episodes": len(episodes),
        "num_sets": num_sets,
        "seed": seed,
        "num_shards": num_shards,
        "store_seq": bool(store_seq),
        "compression": compression,
        "shards": shard_paths,
    }
    with open(os.path.join(split_out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build TFRecord episode shards for FSDiT.")
    parser.add_argument("--data_dir", required=True, help="miniImageNet split root with train/val/test.")
    parser.add_argument("--embeddings_dir", required=True, help="Embedding root with train/val/test .npz.")
    parser.add_argument("--out_dir", required=True, help="Output root for TFRecord shards.")
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits to export.")
    parser.add_argument("--num_sets", type=int, default=100, help="Sets per class (must match training setup).")
    parser.add_argument("--seed", type=int, default=42, help="Episode sampling seed.")
    parser.add_argument("--num_shards", type=int, default=64, help="Number of shards per split.")
    parser.add_argument("--store_seq", type=int, default=1, help="1=store support seq, 0=pooled-only.")
    parser.add_argument("--compression", default="GZIP", choices=["", "GZIP"], help="TFRecord compression.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in split_list:
        build_split(
            split=split,
            data_dir=args.data_dir,
            embeddings_dir=args.embeddings_dir,
            out_dir=args.out_dir,
            num_sets=args.num_sets,
            seed=args.seed,
            num_shards=args.num_shards,
            store_seq=bool(args.store_seq),
            compression=args.compression,
        )
    print("Done.")


if __name__ == "__main__":
    main()
