"""
dataset.py — miniImageNet few-shot episode loader.

Each class → 100 sets of 6 images → 6 rotations (1 target + 5 support).
Stratified sampling ensures balanced class representation per batch.
"""

import os
import threading
from collections import OrderedDict
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
    is_train=True, seed=42, debug_n=0, embedding_root=None, load_support_seq=True,
    npz_cache_size=0, episode_tfrecord_pattern=None, tfrecord_compression_type="",
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

    if episode_tfrecord_pattern:
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

    episodes, class_names = build_episode_table(data_dir, num_sets, seed)
    n_cls = len(class_names)
    n_ep = len(episodes)
    print(f"[Dataset] {data_dir}: {n_cls} classes, {n_ep} episodes")

    if debug_n > 0:
        episodes = episodes[:debug_n]

    episodes = _interleave_by_class(episodes, n_cls, seed + 1)

    # Optional in-process LRU cache to reduce repeated np.load overhead.
    cache = OrderedDict()
    cache_lock = threading.Lock()

    def cached_load_npz(npz_path):
        if npz_cache_size > 0:
            with cache_lock:
                item = cache.get(npz_path)
                if item is not None:
                    cache.move_to_end(npz_path)
                    return item

        data = np.load(npz_path)
        seq = data['seq']
        pooled = data['pooled']
        item = (seq, pooled)

        if npz_cache_size > 0:
            with cache_lock:
                cache[npz_path] = item
                cache.move_to_end(npz_path)
                if len(cache) > npz_cache_size:
                    cache.popitem(last=False)
        return item

    # Build tensor slices
    targets = [e[0] for e in episodes]
    supports_flat = []
    for e in episodes:
        supports_flat.extend(e[1])
    class_ids = [e[2] for e in episodes]

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(targets),
        tf.data.Dataset.from_tensor_slices(tf.reshape(tf.constant(supports_flat), [-1, 5])),
        tf.data.Dataset.from_tensor_slices(tf.constant(class_ids, dtype=tf.int32)),
    ))

    def load_sample(target_path, support_paths, class_id):
        def read_support_npzs(paths_tensor):
            paths = paths_tensor.numpy()
            seq_list = []
            pooled_list = []
            for raw in paths:
                path_str = raw.decode('utf-8')
                if embedding_root:
                    rel = os.path.relpath(path_str, data_dir)
                    npz_path = os.path.join(embedding_root, os.path.splitext(rel)[0] + '.npz')
                else:
                    npz_path = os.path.splitext(path_str)[0] + '.npz'
                if not os.path.exists(npz_path):
                    raise FileNotFoundError(f"Missing precomputed embedding: {npz_path}")
                seq_raw, pooled_raw = cached_load_npz(npz_path)
                if load_support_seq:
                    seq = seq_raw.astype(np.float16)
                else:
                    seq = np.zeros((196, 768), dtype=np.float16)
                pooled = pooled_raw.astype(np.float16)
                seq_list.append(seq)
                pooled_list.append(pooled)
            return np.stack(seq_list, axis=0), np.stack(pooled_list, axis=0)

        target = decode_target(target_path)

        supports_seq, supports_pooled = tf.py_function(
            read_support_npzs,
            [support_paths],
            [tf.float16, tf.float16],
        )
        supports_seq.set_shape([5, 196, 768])
        supports_pooled.set_shape([5, 768])

        return {
            'target': target,
            'supports_seq': supports_seq,
            'supports_pooled': supports_pooled,
            'class_id': class_id,
        }

    ds = ds.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)
    ds = ds.repeat()
    if not debug_n:
        ds = ds.shuffle(min(len(episodes), n_cls * 50), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_names
