#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path('/home/yourslewis/lrm-scaling-all-events')
BASE = ROOT / 'proposed_2-mmoe_ple/config/generated_p23_coordinate_search/p23_page_s10_p09_m01_o00.gin'
TRAIN = ROOT / 'proposed_2-mmoe_ple/train'
DATA = Path('/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2')
EMB = Path('/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings')
OUTROOT = ROOT / 'results_v2'
RUNNER_DIR = OUTROOT / 'p25_uniform_dropout_scaling_autoresearch'
CFGDIR = ROOT / 'proposed_2-mmoe_ple/config/generated_p25_uniform_dropout_scaling'

BASELINE = {
    'p20_s300_page07': {'overall_hr10': 0.6056, 'ads_hr10': 0.2121, 'overall_ndcg10': 0.5590, 'ads_ndcg10': 0.1467},
    'p23_page_s10_p09_m01_o00': {'overall_hr10': 0.6062, 'ads_hr10': 0.2323, 'overall_ndcg10': 0.5521, 'ads_ndcg10': 0.1858},
}

@dataclass(frozen=True)
class Profile:
    name: str
    original_window: int
    keep_rate: float
    model_len: int
    batch: int
    eval_batches: int
    gpu: int
    port: int

PROFILES = [
    Profile('p25_w0100_drop50_s10_p09_m01_o00', 100, 0.5, 50, 16, 100, 0, 30530),
    Profile('p25_w0200_drop50_s10_p09_m01_o00', 200, 0.5, 100, 16, 100, 1, 30531),
    Profile('p25_w0500_drop50_s10_p09_m01_o00', 500, 0.5, 250, 16, 100, 0, 30532),
    Profile('p25_w1000_drop50_s10_p09_m01_o00', 1000, 0.5, 500, 16, 100, 1, 30533),
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    (RUNNER_DIR / 'logs').mkdir(parents=True, exist_ok=True)
    CFGDIR.mkdir(parents=True, exist_ok=True)


def score(overall: Optional[float], ads: Optional[float]) -> Optional[float]:
    if overall is None or ads is None:
        return None
    return 0.4 * float(overall) + 0.6 * float(ads)


def patch_config(p: Profile) -> str:
    text = BASE.read_text()
    replacements = {
        'get_reco_dataset.max_sequence_length = 200': f'get_reco_dataset.max_sequence_length = {p.model_len}',
        'Trainer.local_batch_size = 32': f'Trainer.local_batch_size = {p.batch}',
        'Trainer.eval_batch_size = 32': f'Trainer.eval_batch_size = {p.batch}',
        'Trainer.eval_max_batches = 50': f'Trainer.eval_max_batches = {p.eval_batches}',
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f'Missing expected config line in {BASE}: {old}')
        text = text.replace(old, new)
    insert = (
        f'get_reco_dataset.original_sequence_length = {p.original_window}\n'
        f'get_reco_dataset.history_keep_rate = {p.keep_rate:.3f}\n'
        f'get_reco_dataset.history_sample_seed = 20260514\n'
    )
    marker = f'get_reco_dataset.max_sequence_length = {p.model_len}\n'
    text = text.replace(marker, marker + insert)
    header = (
        f'# P25 uniform history-dropout scaling experiment: {p.name}\n'
        f'# original_window={p.original_window}, keep_rate={p.keep_rate}, '
        f'model_len={p.model_len}, batch={p.batch}\n\n'
    )
    return header + text


def write_configs() -> list[Path]:
    ensure_dirs()
    paths = []
    for p in PROFILES:
        cfg = CFGDIR / f'{p.name}.gin'
        cfg.write_text(patch_config(p))
        paths.append(cfg)
    return paths


def launch(p: Profile) -> tuple[subprocess.Popen, Path, Path, Path]:
    cfg = CFGDIR / f'{p.name}.gin'
    if not cfg.exists():
        cfg.write_text(patch_config(p))
    out = OUTROOT / p.name
    logdir = out / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f'train_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.log'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(p.gpu)
    env['AZUREML_RUN_ID'] = f'{p.name}_{datetime.now(timezone.utc).strftime("%Y%m%d")}'
    env['PYTHONPATH'] = f'{TRAIN}:{env.get("PYTHONPATH", "")}'
    cmd = [
        'torchrun', '--nproc_per_node=1', f'--master_port={p.port}', 'main.py',
        f'--gin_config_file={cfg}', f'--output_path={out}', '--data_path', str(DATA), '--mode=job',
        '--ads_semantic_embd_path', str(EMB / 'domain_0'),
        '--web_browsing_semantic_embd_path', str(EMB / 'domain_1'),
        '--shopping_semantic_embd_path', str(EMB / 'domain_2'),
        '--ads_pure_corpus_embd_path', str(EMB / 'domain_3'),
        '--other_semantic_embd_path', str(EMB / 'domain_4'),
    ]
    f = open(log, 'w')
    proc = subprocess.Popen(cmd, cwd=TRAIN, env=env, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    (out / 'train.pid').write_text(str(proc.pid))
    return proc, out, log, cfg


def monitor(out: Path) -> Optional[dict]:
    files = sorted(out.glob('*/validation_monitor.json'), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text())
    except Exception:
        return None


def metrics(entry: Optional[dict]) -> dict:
    if not entry:
        return {}
    m = entry.get('metrics') or {}
    return {
        'batch': entry.get('batch'),
        'overall_ndcg10': entry.get('value') if entry.get('value') is not None else m.get('ndcg_10'),
        'overall_hr10': m.get('hr_10'),
        'ads_hr10': m.get('ads_hr_10'),
        'ads_ndcg10': m.get('ads_ndcg_10'),
    }


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass
    time.sleep(8)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def has_bad_log(log: Path) -> Optional[str]:
    if not log.exists():
        return None
    try:
        tail = log.read_text(errors='ignore')[-20000:].lower()
    except Exception:
        return None
    # Ignore SIGTERM tracebacks emitted by torchrun after this runner intentionally stops at max_batch.
    if 'signalexception' in tail and 'got signal: 15' in tail:
        return None
    for needle in ['out of memory', 'cuda error', 'runtimeerror', 'childfailederror']:
        if needle in tail:
            return needle
    return None


def run_one(p: Profile, max_batch: int, poll_seconds: int) -> dict:
    proc, out, log, cfg = launch(p)
    print(f'START {p.name} W={p.original_window} keep={p.keep_rate} model_len={p.model_len} batch={p.batch} gpu={p.gpu} pid={proc.pid}', flush=True)
    reason = 'process exited'
    best_score = None
    best_ads = None
    best_overall = None
    while proc.poll() is None:
        time.sleep(poll_seconds)
        bad = has_bad_log(log)
        if bad:
            reason = f'log failure: {bad}'
            terminate(proc)
            break
        mon = monitor(out)
        latest = metrics((mon or {}).get('latest'))
        batch = latest.get('batch') or 0
        overall = latest.get('overall_hr10')
        ads = latest.get('ads_hr10')
        sc = score(overall, ads)
        if sc is not None:
            best_score = sc if best_score is None else max(best_score, sc)
        if ads is not None:
            best_ads = ads if best_ads is None else max(best_ads, ads)
        if overall is not None:
            best_overall = overall if best_overall is None else max(best_overall, overall)
        print(f'{p.name} batch={batch} overall={overall} ads={ads} score={sc}', flush=True)
        if batch >= max_batch:
            reason = f'reached {max_batch}'
            terminate(proc)
            break
    mon = monitor(out)
    latest = metrics((mon or {}).get('latest'))
    best = metrics((mon or {}).get('best'))
    rec = {
        'name': p.name,
        'profile': asdict(p),
        'reason': reason,
        'returncode': proc.poll(),
        'latest': latest,
        'best_by_ndcg': best,
        'latest_score': score(latest.get('overall_hr10'), latest.get('ads_hr10')),
        'best_score_seen': best_score,
        'best_ads_hr10_seen': best_ads,
        'best_overall_hr10_seen': best_overall,
        'passes_guardrail': bool((latest.get('overall_hr10') or 0) >= 0.58 and (latest.get('ads_hr10') or 0) >= 0.2121),
        'config': str(cfg),
        'out': str(out),
        'log': str(log),
        'ended_at': utc(),
    }
    with (RUNNER_DIR / 'summary.jsonl').open('a') as f:
        f.write(json.dumps(rec) + '\n')
    print(f'END {p.name}: {reason}', flush=True)
    return rec


def run_wave(profiles: list[Profile], max_batch: int, poll_seconds: int) -> list[dict]:
    results = []
    lock = threading.Lock()
    def worker(p: Profile):
        r = run_one(p, max_batch=max_batch, poll_seconds=poll_seconds)
        with lock:
            results.append(r)
    threads = [threading.Thread(target=worker, args=(p,)) for p in profiles]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--generate-only', action='store_true')
    ap.add_argument('--launch', action='store_true')
    ap.add_argument('--max-batch', type=int, default=26000)
    ap.add_argument('--poll-seconds', type=int, default=60)
    ap.add_argument('--windows', default='100,200,500,1000', help='Comma-separated original windows to generate/launch')
    args = ap.parse_args()
    ensure_dirs()
    wanted = {int(x) for x in args.windows.split(',') if x.strip()}
    selected = [p for p in PROFILES if p.original_window in wanted]
    paths = write_configs()
    print('Generated configs:')
    for p in selected:
        print(f'  {CFGDIR / (p.name + ".gin")}')
    (RUNNER_DIR / 'state.json').write_text(json.dumps({
        'created_at': utc(),
        'profiles': [asdict(p) for p in selected],
        'baseline': BASELINE,
        'max_batch': args.max_batch,
        'status': 'generated' if not args.launch else 'running',
    }, indent=2))
    if args.generate_only or not args.launch:
        return
    all_results = []
    all_results += run_wave([p for p in selected if p.original_window in (100, 200)], args.max_batch, args.poll_seconds)
    all_results += run_wave([p for p in selected if p.original_window in (500, 1000)], args.max_batch, args.poll_seconds)
    (RUNNER_DIR / 'state.json').write_text(json.dumps({
        'created_at': utc(),
        'profiles': [asdict(p) for p in selected],
        'baseline': BASELINE,
        'max_batch': args.max_batch,
        'status': 'all_done',
        'results': all_results,
    }, indent=2))
    print('ALL_DONE', flush=True)

if __name__ == '__main__':
    main()
