#!/usr/bin/env python3
"""
Step 1 (v3): MEMORY-BOUNDED vocab builder via two-pass hash bucketing.

Why v3: step1_v2 held the ENTIRE per-domain {text: id} dict in RAM (plus an
inverted {id: text}). On newdata-v4 (~150-230M unique texts) that OOM-kills even
a 140 GB box. v3 never holds the full vocabulary in memory.

Design (exact, not approximate):
  PASS 1 (spill):  stream every train/val shard, normalize text exactly like
                   step1_v2 (extract_text_normalized), and APPEND each
                   (domain, text) to a bucket file chosen by a stable hash of
                   the text. NB = number of buckets per domain. Each bucket is a
                   newline-delimited text file. Memory here is O(write buffers),
                   independent of vocabulary size.
  PASS 2 (dedupe): process ONE (domain, bucket) at a time. Load just that
                   bucket's lines into a set, dedupe, and assign IDs. Memory peak
                   = largest single bucket (≈ total_unique / NB). Determinism is
                   preserved by assigning IDs in a fixed order: buckets ascending,
                   and within a bucket the texts sorted lexicographically. IDs are
                   contiguous starting at MIN_ITEM_ID, identical layout to v2's
                   sequential scheme as long as downstream only requires a stable
                   bijection text<->id (encode/step3 both do).

Outputs (same names + semantics as step1_v2, so step3/encode are compatible):
  <out>/domain_<d>_text2id.pkl   {text: id}
  <out>/domain_<d>_id2text.pkl   {id: text}
  <out>/vocab_meta.json          {min_item_id, domains{d:{num_unique_texts,
                                  max_item_id, shard_size}}, total_unique_texts}

NOTE on downstream memory: step1_v2 wrote ONE giant pickle per domain, and
step3_v2 then loaded all 5 at once -> step3 ALSO OOMs at this scale. v3 ALSO emits
an optional sharded, mmap-friendly mapping (domain_<d>_text2id/ dir of bucket
pickles + a manifest) so a v3-aware step3 can look up by the SAME hash without
loading the whole dict. The monolithic .pkl is still written for backward compat;
pass --no_monolithic to skip it when the full dict would not fit in the consumer.
"""

# Workflow notes:
# Bounded-memory v3 vocab builder for the serial CPU path: scan train/val events,
# normalize text by domain, spill domain/bucket text, reduce each bucket, and
# write sharded text2id/id2text vocab artifacts.
# Performance tricks:
# - Hash-bucket spills cap peak memory and allow very large vocabularies.
# - LRU file handles avoid keeping 5*num_buckets descriptors open at once.
# - --no_monolithic skips huge pickle maps when downstream scripts can read shards.

import argparse, json, glob, os, logging, pickle, shutil
from collections import OrderedDict

try:
    from data_prep.vocab_common_v3 import EVENT_TO_DOMAIN, MIN_ITEM_ID, NUM_DOMAINS, bucket_of, extract_text_normalized
except ModuleNotFoundError:  # pragma: no cover - supports `python data_prep/step1_collect_vocab_v3.py`
    from vocab_common_v3 import EVENT_TO_DOMAIN, MIN_ITEM_ID, NUM_DOMAINS, bucket_of, extract_text_normalized

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ------------------------------ orjson optional ------------------------------
try:
    import orjson as _json
    def loads(b):  # bytes or str
        return _json.loads(b)
except Exception:  # pragma: no cover
    import json as _json
    def loads(b):
        return _json.loads(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_buckets", type=int, default=256,
                   help="Buckets per domain. Peak RAM in pass 2 ~= total_unique/num_buckets.")
    p.add_argument("--spill_dir", default=None,
                   help="Where to write bucket spill files (default: a tmp dir under output_dir).")
    p.add_argument("--max_open_handles", type=int, default=128,
                   help="Max spill files kept open at once (prevents fd-limit issues).")
    p.add_argument("--no_monolithic", action="store_true",
                   help="Skip the giant domain_<d>_text2id.pkl / id2text.pkl (write sharded only).")
    p.add_argument("--keep_spill", action="store_true", help="Do not delete spill files (debug).")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    NB = args.num_buckets
    spill_root = args.spill_dir or os.path.join(args.output_dir, "_spill")
    os.makedirs(spill_root, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(args.train_dir, "train_chunk_*.tsv")))
    if args.val_dir:
        all_files += sorted(glob.glob(os.path.join(args.val_dir, "val_chunk_*.tsv")))
    logging.info(f"[pass1] scanning {len(all_files)} files -> {NUM_DOMAINS} domains x {NB} buckets")

    # ---- PASS 1: spill (domain,text) into bucket append-files ----
    # Bounded LRU of open handles. Keeping NUM_DOMAINS*NB handles open can exceed
    # fd limits; opening per event is too slow. This keeps writes fast and safe.
    handles = OrderedDict()
    def hfor(d, b):
        key = (d, b)
        h = handles.get(key)
        if h is None:
            if len(handles) >= args.max_open_handles:
                _, old = handles.popitem(last=False)
                old.close()
            dpath = os.path.join(spill_root, f"domain_{d}")
            os.makedirs(dpath, exist_ok=True)
            h = open(os.path.join(dpath, f"bucket_{b:04d}.txt"), "a",
                     encoding="utf-8", errors="surrogatepass")
            handles[key] = h
        else:
            handles.move_to_end(key)
        return h

    n_events = 0
    for fi, fpath in enumerate(all_files):
        with open(fpath, encoding="utf-8", errors="surrogatepass") as f:
            try:
                next(f)  # header
            except StopIteration:
                continue
            for line in f:
                tab = line.find("\t")
                if tab < 0:
                    continue
                payload = line[tab + 1:].rstrip("\n")
                try:
                    events = loads(payload)
                except Exception:
                    continue
                for ev in events:
                    etype = ev.get("Type", "")
                    d = EVENT_TO_DOMAIN.get(etype)
                    if d is None:
                        continue
                    text = extract_text_normalized(ev)
                    b = bucket_of(text, NB)
                    # newline-delimited; texts may contain no newline (events are single-line JSON)
                    hfor(d, b).write(text.replace("\n", " ") + "\n")
                    n_events += 1
        if (fi + 1) % 10 == 0 or fi == len(all_files) - 1:
            logging.info(f"[pass1] {fi+1}/{len(all_files)} files, {n_events:,} events spilled")
    for h in handles.values():
        h.close()
    logging.info(f"[pass1] done: {n_events:,} events spilled to {spill_root}")

    # ---- PASS 2: per (domain,bucket) dedupe + sequential id assignment ----
    meta_domains = {}
    total_texts = 0
    for d in range(NUM_DOMAINS):
        dpath = os.path.join(spill_root, f"domain_{d}")
        next_id = MIN_ITEM_ID
        # Sharded output dirs. step3_v3 reads text2id shards; encode_v3 reads
        # id2text shards. Neither downstream step has to load the full vocab.
        text2id_dir = os.path.join(args.output_dir, f"domain_{d}_text2id")
        id2text_dir = os.path.join(args.output_dir, f"domain_{d}_id2text")
        os.makedirs(text2id_dir, exist_ok=True)
        os.makedirs(id2text_dir, exist_ok=True)
        manifest = {"num_buckets": NB, "min_item_id": MIN_ITEM_ID, "buckets": {}}

        mono_text2id = {} if not args.no_monolithic else None
        mono_id2text = {} if not args.no_monolithic else None

        for b in range(NB):
            bpath = os.path.join(dpath, f"bucket_{b:04d}.txt")
            uniq = set()
            if os.path.exists(bpath):
                with open(bpath, encoding="utf-8", errors="surrogatepass") as f:
                    for line in f:
                        uniq.add(line.rstrip("\n"))
            if not uniq:
                manifest["buckets"][b] = {"start_id": next_id, "count": 0}
                continue
            # deterministic order within bucket
            ordered = sorted(uniq)
            start_id = next_id
            bucket_map = {}
            for t in ordered:
                bucket_map[t] = next_id
                next_id += 1
            bucket_i2t = {i: t for t, i in bucket_map.items()}
            # write this bucket's mappings (small)
            with open(os.path.join(text2id_dir, f"bucket_{b:04d}.pkl"), "wb") as f:
                pickle.dump(bucket_map, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(id2text_dir, f"bucket_{b:04d}.pkl"), "wb") as f:
                pickle.dump(bucket_i2t, f, protocol=pickle.HIGHEST_PROTOCOL)
            manifest["buckets"][b] = {"start_id": start_id, "count": len(ordered)}
            if mono_text2id is not None:
                mono_text2id.update(bucket_map)
                mono_id2text.update(bucket_i2t)
            del uniq, ordered, bucket_map, bucket_i2t

        n_unique = next_id - MIN_ITEM_ID
        total_texts += n_unique
        with open(os.path.join(text2id_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        with open(os.path.join(id2text_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        if mono_text2id is not None:
            with open(os.path.join(args.output_dir, f"domain_{d}_text2id.pkl"), "wb") as f:
                pickle.dump(mono_text2id, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(args.output_dir, f"domain_{d}_id2text.pkl"), "wb") as f:
                pickle.dump(mono_id2text, f, protocol=pickle.HIGHEST_PROTOCOL)
            del mono_text2id, mono_id2text
        meta_domains[str(d)] = {
            "num_unique_texts": n_unique,
            "max_item_id": next_id - 1,
            "shard_size": n_unique + MIN_ITEM_ID,
            "num_buckets": NB,
        }
        logging.info(f"[pass2] domain {d}: {n_unique:,} unique texts, ids [{MIN_ITEM_ID},{next_id-1}]")

    meta = {
        "min_item_id": MIN_ITEM_ID,
        "url_normalized": True,
        "builder": "step1_v3_hashbucket",
        "num_buckets": NB,
        "domains": meta_domains,
        "total_unique_texts": total_texts,
    }
    with open(os.path.join(args.output_dir, "vocab_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"[done] TOTAL {total_texts:,} unique texts; meta written")

    if not args.keep_spill:
        shutil.rmtree(spill_root, ignore_errors=True)
        logging.info(f"[cleanup] removed spill dir {spill_root}")


if __name__ == "__main__":
    main()
