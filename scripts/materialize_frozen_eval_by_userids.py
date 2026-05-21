#!/usr/bin/env python3
"""Materialize a frozen eval subset from an already processed evalsplit.

This utility selects raw validation shards by numeric chunk id, reads their user_ids,
filters an already processed evalsplit by those user_ids, and writes a stable parquet
subset split into fixed-size parts. It avoids re-running the expensive vocab-based
conversion when the source processed dataset already has miss_count == 0.

Example:
  python scripts/materialize_frozen_eval_by_userids.py \
    --raw-val-dir /home/yourslewis/lrm_benchmarkv4/val \
    --source-processed /home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_evalsplit \
    --output-dir /home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_eval7raw_0_6 \
    --chunk-ids 0,1,2,3,4,5,6
"""
import argparse
import json
import pathlib
import shutil
import time

import pyarrow as pa
import pyarrow.parquet as pq

DOMAIN_OFFSET = 1_000_000_000


def parse_chunk_ids(text: str) -> list[int]:
    ids: list[int] = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--raw-val-dir', required=True, type=pathlib.Path)
    p.add_argument('--source-processed', required=True, type=pathlib.Path,
                   help='Processed dataset dir with metadata.json and eval/part_*.parquet')
    p.add_argument('--output-dir', required=True, type=pathlib.Path)
    p.add_argument('--chunk-ids', default='0-6', help='Comma/range list, e.g. 0-6 or 0,1,2')
    p.add_argument('--part-size', type=int, default=5000)
    args = p.parse_args()

    chunk_ids = parse_chunk_ids(args.chunk_ids)
    out_eval = args.output_dir / 'eval'
    out_eval.mkdir(parents=True, exist_ok=True)
    for old in out_eval.glob('part_*.parquet'):
        old.unlink()

    selected_users: set[str] = set()
    raw_line_counts: dict[str, int] = {}
    for cid in chunk_ids:
        raw_path = args.raw_val_dir / f'val_chunk_{cid}.tsv'
        n = 0
        with raw_path.open() as f:
            next(f)
            for line in f:
                selected_users.add(line.split('\t', 1)[0])
                n += 1
        raw_line_counts[raw_path.name] = n

    source_meta_path = args.source_processed / 'metadata.json'
    meta = json.loads(source_meta_path.read_text())
    if meta.get('miss_count') not in (0, None):
        raise ValueError(f"source processed dataset has miss_count={meta.get('miss_count')}; refusing shortcut")

    buffer: list[pa.Table] = []
    buffer_rows = 0
    out_index = 0
    total = 0
    domain_counts: dict[str, int] = {}

    def flush(force: bool = False) -> None:
        nonlocal buffer, buffer_rows, out_index
        while buffer_rows >= args.part_size or (force and buffer_rows > 0):
            need = buffer_rows if force and buffer_rows < args.part_size else args.part_size
            pieces: list[pa.Table] = []
            taken = 0
            new_buffer: list[pa.Table] = []
            new_rows = 0
            for table in buffer:
                if taken >= need:
                    new_buffer.append(table)
                    new_rows += table.num_rows
                    continue
                rem = need - taken
                if table.num_rows <= rem:
                    pieces.append(table)
                    taken += table.num_rows
                else:
                    pieces.append(table.slice(0, rem))
                    new_buffer.append(table.slice(rem))
                    new_rows += table.num_rows - rem
                    taken += rem
            pq.write_table(pa.concat_tables(pieces), out_eval / f'part_{out_index:04d}.parquet')
            out_index += 1
            buffer = new_buffer
            buffer_rows = new_rows

    for part in sorted((args.source_processed / 'eval').glob('part_*.parquet')):
        table = pq.read_table(part)
        user_ids = table.column('user_id').to_pylist()
        idx = [i for i, uid in enumerate(user_ids) if uid in selected_users]
        if not idx:
            continue
        sub = table.take(pa.array(idx, type=pa.int64()))
        for encoded_ids in sub.column('encoded_ids').to_pylist():
            domain = int(encoded_ids[-1]) // DOMAIN_OFFSET
            domain_counts[str(domain)] = domain_counts.get(str(domain), 0) + 1
        buffer.append(sub)
        buffer_rows += sub.num_rows
        total += sub.num_rows
        flush(False)
    flush(True)

    train_link = args.output_dir / 'train'
    if train_link.exists() or train_link.is_symlink():
        if train_link.is_symlink() or train_link.is_file():
            train_link.unlink()
        else:
            shutil.rmtree(train_link)
    train_link.symlink_to(args.source_processed / 'train', target_is_directory=True)

    meta['num_eval_users'] = total
    meta['frozen_eval_manifest'] = str(args.output_dir / 'freeze_manifest.json')
    (args.output_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, sort_keys=True))

    manifest = {
        'name': args.output_dir.name,
        'source_processed_eval': str(args.source_processed / 'eval'),
        'source_raw_val_dir': str(args.raw_val_dir),
        'selected_raw_val_chunks': [f'val_chunk_{i}.tsv' for i in chunk_ids],
        'selection_rule': 'deterministic raw validation chunk ids; selected by raw user_id membership',
        'construction': 'filter processed evalsplit parts by user_id from selected raw shards',
        'raw_line_counts': raw_line_counts,
        'selected_raw_unique_users': len(selected_users),
        'num_eval_users': total,
        'split_parts': out_index,
        'target_domain_counts': domain_counts,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    (args.output_dir / 'freeze_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
