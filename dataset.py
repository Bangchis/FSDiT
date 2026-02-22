"""
dataset.py — miniImageNet few-shot episode loader.

Each class → 100 sets of 6 images → 6 rotations (1 target + 5 support).
Stratified sampling ensures balanced class representation per batch.
"""

import os
import numpy as np
import tensorflow as tf


def build_episode_table(data_dir, num_sets=100, seed=42):
    """
    Scan class folders, generate all (target, supports, class_id) episodes.

    Returns:
        episodes: list of (target_path, [5 support_paths], class_idx)
        class_names: sorted list of class folder names
    """
    rng = np.random.RandomState(seed)
    class_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    episodes = []
    for cls_idx, cls_name in enumerate(class_dirs):
        cls_path = os.path.join(data_dir, cls_name)
        imgs = sorted(
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))
        )
        assert len(imgs) >= 6, f"Class '{cls_name}' has {len(imgs)} images (need ≥ 6)"

        for _ in range(num_sets):
            chosen = [imgs[i] for i in rng.choice(len(imgs), 6, replace=False)]
            for rot in range(6):
                target = chosen[rot]
                supports = [chosen[j] for j in range(6) if j != rot]
                episodes.append((target, supports, cls_idx))

    return episodes, class_dirs


def _interleave_by_class(episodes, num_classes, seed):
    """Round-robin interleave episodes across classes for balanced batching."""
    rng = np.random.RandomState(seed)
    buckets = {c: [] for c in range(num_classes)}
    for ep in episodes:
        buckets[ep[2]].append(ep)
    for c in range(num_classes):
        rng.shuffle(buckets[c])

    result = []
    max_len = max(len(v) for v in buckets.values())
    for i in range(max_len):
        for c in range(num_classes):
            if i < len(buckets[c]):
                result.append(buckets[c][i])
    return result


def build_dataset(
    data_dir, batch_size, image_size=224, num_sets=100,
    is_train=True, seed=42, debug_n=0, load_support_seq=True,
    episode_tfrecord_pattern=None, tfrecord_compression_type="",
):
    """
    Build tf.data pipeline for FSDiT training.

    Returns:
        dataset: yields {
            'target': (B,H,W,3),
            'supports_seq': (B,5,196,768),
            'supports_pooled': (B,5,768),
            'class_id': (B,)
        }
        class_names: list of class names
    """
    def decode_target(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [image_size, image_size])
        img = tf.cast(img, tf.float32) / 255.0
        img = (img - 0.5) / 0.5  # [-1, 1]
        if is_train:
            img = tf.image.random_flip_left_right(img)
        return img

    if not episode_tfrecord_pattern:
        raise ValueError(
            "TFRecord-only mode: please pass `episode_tfrecord_pattern` to build_dataset(). "
            "Legacy npz runtime loader was removed for maintainability."
        )

    files = tf.io.gfile.glob(episode_tfrecord_pattern)
    if not files:
        raise FileNotFoundError(f"No TFRecord files matched pattern: {episode_tfrecord_pattern}")
    print(f"[Dataset] TFRecord mode: {len(files)} shards from {episode_tfrecord_pattern}")

    ds = tf.data.TFRecordDataset(
        files,
        compression_type=tfrecord_compression_type or None,
        num_parallel_reads=tf.data.AUTOTUNE,
    )

    feature_spec = {
        'target_path': tf.io.FixedLenFeature([], tf.string),
        'class_id': tf.io.FixedLenFeature([], tf.int64),
        'supports_pooled': tf.io.FixedLenFeature([], tf.string),
        'supports_seq': tf.io.FixedLenFeature([], tf.string, default_value=b''),
    }

    def parse_example(example_proto):
        ex = tf.io.parse_single_example(example_proto, feature_spec)
        target = decode_target(ex['target_path'])
        supports_pooled = tf.io.decode_raw(ex['supports_pooled'], tf.float16)
        supports_pooled = tf.reshape(supports_pooled, [5, 768])

        if load_support_seq:
            has_seq = tf.greater(tf.strings.length(ex['supports_seq']), 0)
            supports_seq = tf.cond(
                has_seq,
                lambda: tf.reshape(tf.io.decode_raw(ex['supports_seq'], tf.float16), [5, 196, 768]),
                lambda: tf.zeros([5, 196, 768], dtype=tf.float16),
            )
        else:
            supports_seq = tf.zeros([5, 196, 768], dtype=tf.float16)

        return {
            'target': target,
            'supports_seq': supports_seq,
            'supports_pooled': supports_pooled,
            'class_id': tf.cast(ex['class_id'], tf.int32),
        }

    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)
    ds = ds.repeat()
    if not debug_n:
        ds = ds.shuffle(8192, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, []
