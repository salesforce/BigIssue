import itertools
import math

from pathlib import Path
import random

import tensorflow as tf


def create_resumable_iter(files, batch_size, seq_len, resume_at_file=None, resume_at_batch=None):
    def iterator_from_tfrecords_files(files, seq_len, batch_size):

        def parse_fn(sample):
            parsed_features = tf.io.parse_single_example(sample, {'text': tf.io.VarLenFeature(tf.int64)})
            return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features['text'])), tf.uint32)

        ds = tf.data.TFRecordDataset(files)
        ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        for batch in ds:
            yield batch.numpy()

    def resume_to_file(files_iter, resume_at_file=None):
        f = next(files_iter)
        print(f'Skipping from {f} to file {resume_at_file}')
        while f != resume_at_file:
            f = next(files_iter)
        print(f'Skipped to file {f}')
        assert f == resume_at_file
        return files_iter, f

    def group_iter(iter, group_size):
        # assumes never ending iter
        list = []
        while True:
            if len(list) == group_size:
                yield list
                list = []
            list.append(next(iter))

    files_iter = group_iter(iter(itertools.cycle(files)), group_size=batch_size)

    if resume_at_file is not None:
        files_iter, f = resume_to_file(files_iter, resume_at_file)
        batch_iter = iterator_from_tfrecords_files(f, seq_len=seq_len, batch_size=batch_size)
        for b, records in enumerate(batch_iter):
            if b < resume_at_batch:
                continue
            yield (records, f, b)

    for f in files_iter:
        batch_iter = iterator_from_tfrecords_files(f, seq_len=seq_len, batch_size=batch_size)
        for b, records in enumerate(batch_iter):
            yield (records, f, b)


def list_files(path):
    is_gcs_path = path.startswith('gs://')
    filenames = tf.io.gfile.glob(path) if is_gcs_path else [str(p) for p in Path(path).glob(path)]
    return sorted(filenames)


def create_resumable_partition_iter(path, batch_size, seq_len, process_id, process_n, resume_at_file=None,
                                    resume_at_batch=None):
    files = list_files(path)
    files_partition = [f for (i, f) in enumerate(files) if i % process_n == process_id]
    return create_resumable_iter(files=files_partition, batch_size=batch_size, seq_len=seq_len,
                                 resume_at_file=resume_at_file, resume_at_batch=resume_at_batch), files_partition


def create_resumable_partition_iter_split(path, batch_size, seq_len, process_id, process_n, resume_at_file=None,
                                          resume_at_batch=None, ratio=0.9):
    files = sorted(list_files(path))
    random.shuffle(files)
    print(files[0])
    print(files[-1])
    boundary = int(len(files) * ratio)
    train_files, test_files = files[:boundary], files[boundary:]
    files_partition = [f for (i, f) in enumerate(train_files) if i % process_n == process_id]
    train_iter = create_resumable_iter(files=files_partition, batch_size=batch_size, seq_len=seq_len,
                                       resume_at_file=resume_at_file, resume_at_batch=resume_at_batch)
    test_iter = create_resumable_iter(files=test_files, batch_size=batch_size, seq_len=seq_len)
    return train_iter, files_partition, test_iter, math.ceil(len(test_files) / batch_size)
