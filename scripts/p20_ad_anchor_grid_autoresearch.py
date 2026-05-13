#!/usr/bin/env python3
"""Run a small P20 ad-anchor grid with early stopping.

Grid dimensions:
- ad_anchor_sigma_seconds: decay sharpness
- ad_anchor_page_title_gate: EdgePageTitle/ChromePageTitle event-type gate

The script launches one run at a time on the requested GPU, watches
validation_monitor.json, and terminates early when the run is clearly behind.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/yourslewis/lrm-scaling-all-events')
BASE_CONFIG = ROOT / 'proposed_2-mmoe_ple/config/proposed19b_sharp_ad_anchor_event_gate.gin'
TRAIN_DIR = ROOT / 'proposed_2-mmoe_ple/train'
DATA = Path('/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2')
EMB = Path('/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings')
RESULTS = ROOT / 'results_v2'
GRID_DIR = ROOT / 'results_v2/p20_ad_anchor_grid_autoresearch'
CONFIG_DIR = ROOT / 'proposed_2-mmoe_ple/config/generated_p20_grid'

# Keep this focused: two dimensions only.
GRID = [
    # name, sigma_seconds, page_title_gate
    ('p20_s180_page07', 180.0, 0.7),
    ('p20_s180_page10', 180.0, 1.0),
    ('p20_s240_page07', 240.0, 0.7),
    ('p20_s240_page10', 240.0, 1.0),
    ('p20_s300_page07', 300.0, 0.7),
    ('p20_s300_page10', 300.0, 1.0),
]

BASELINES = {
    'p18_best_ads_hr10': 0.1744,
    'p19b_best_ads_hr10': 0.1634,
}


def patch_config(text: str, sigma: float, page_gate: float, name: str) -> str:
    text = text.replace('proposed19b_sharp_ad_anchor_event_gate', name)
    replacements = {
        'make_model.ad_anchor_sigma_seconds = 180.0': f'make_model.ad_anchor_sigma_seconds = {sigma:.1f}',
        'make_model.ad_anchor_event_gate_mode = "event_type"': (
            'make_model.ad_anchor_event_gate_mode = "event_type"\n'
            'make_model.ad_anchor_strong_event_gate = 1.0\n'
            f'make_model.ad_anchor_page_title_gate = {page_gate:.3f}\n'
            'make_model.ad_anchor_msn_gate = 0.1'
        ),
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f'Config pattern not found: {old}')
        text = text.replace(old, new)
    text = text.replace(
        '# P19B: sharp bidirectional ad-anchor weighting with event-type gate.',
        f'# P20 grid: sigma={sigma:.0f}s, Edge/Chrome PageTitle gate={page_gate:.3f}.',
    )
    return text


def load_monitor(out_path: Path, name: str) -> dict | None:
    p = out_path / f'{name}_20260513' / 'validation_monitor.json'
    if not p.exists():
        matches = sorted(out_path.glob('*/validation_monitor.json'), key=lambda x: x.stat().st_mtime, reverse=True)
        p = matches[0] if matches else p
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def get_latest_best(monitor: dict | None) -> tuple[dict, dict]:
    if not monitor:
        return {}, {}
    return monitor.get('latest') or {}, monitor.get('best') or {}


def terminate(proc: subprocess.Popen, grace: int = 20) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def launch_run(name: str, config_path: Path, out_path: Path, gpu: int, port: int, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['AZUREML_RUN_ID'] = f'{name}_{datetime.now(timezone.utc).strftime("%Y%m%d")}'
    env['PYTHONPATH'] = f'{TRAIN_DIR}:{env.get("PYTHONPATH", "")}'
    cmd = [
        'torchrun', '--nproc_per_node=1', f'--master_port={port}', 'main.py',
        f'--gin_config_file={config_path}',
        f'--output_path={out_path}',
        '--data_path', str(DATA), '--mode=job',
        '--ads_semantic_embd_path', str(EMB / 'domain_0'),
        '--web_browsing_semantic_embd_path', str(EMB / 'domain_1'),
        '--shopping_semantic_embd_path', str(EMB / 'domain_2'),
        '--ads_pure_corpus_embd_path', str(EMB / 'domain_3'),
        '--other_semantic_embd_path', str(EMB / 'domain_4'),
    ]
    log_f = open(log_path, 'w')
    return subprocess.Popen(cmd, cwd=TRAIN_DIR, env=env, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def should_stop(latest: dict, started_at: float, max_batches: int, patience_evals: int, state: dict) -> tuple[bool, str]:
    batch = int(latest.get('batch') or 0)
    metrics = latest.get('metrics') or {}
    ads_hr = metrics.get('ads_hr_10')
    overall_hr = metrics.get('hr_10')
    if batch >= max_batches:
        return True, f'reached max_batches={max_batches}'
    if ads_hr is not None:
        state['best_ads_hr10'] = max(float(ads_hr), float(state.get('best_ads_hr10', 0.0)))
    if overall_hr is not None:
        state['best_overall_hr10'] = max(float(overall_hr), float(state.get('best_overall_hr10', 0.0)))

    # Need at least several evals before judging.
    if batch < 10000 or ads_hr is None:
        return False, ''
    evals = batch // 1000
    best_ads = float(state.get('best_ads_hr10', 0.0))
    # Very poor early Ads signal: stop quickly.
    if batch >= 12000 and best_ads < 0.145:
        return True, f'early reject: best_ads_hr10={best_ads:.4f}<0.145 by batch {batch}'
    # If we already crossed P18 best, keep until max_batches for better checkpoint.
    if best_ads >= BASELINES['p18_best_ads_hr10']:
        return False, ''
    # If after 20k still below P19B, unlikely to beat P18.
    if batch >= 20000 and best_ads < BASELINES['p19b_best_ads_hr10']:
        return True, f'reject: best_ads_hr10={best_ads:.4f}<P19B by batch {batch}'
    return False, ''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--port', type=int, default=30220)
    ap.add_argument('--max-batches', type=int, default=26000)
    ap.add_argument('--poll-seconds', type=int, default=60)
    ap.add_argument('--only', nargs='*', default=None)
    args = ap.parse_args()

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = GRID_DIR / 'summary.jsonl'
    base_text = BASE_CONFIG.read_text()

    for idx, (name, sigma, page_gate) in enumerate(GRID):
        if args.only and name not in args.only:
            continue
        out_path = RESULTS / name
        log_dir = out_path / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        config_path = CONFIG_DIR / f'{name}.gin'
        config_path.write_text(patch_config(base_text, sigma, page_gate, name))
        log_path = log_dir / f'autoresearch_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.log'
        state = {'name': name, 'sigma': sigma, 'page_title_gate': page_gate, 'started_at': time.time()}
        print(f'[{datetime.now()}] START {name} sigma={sigma} page_gate={page_gate} gpu={args.gpu}', flush=True)
        proc = launch_run(name, config_path, out_path, args.gpu, args.port + idx, log_path)
        (out_path / 'train.pid').write_text(str(proc.pid))
        stop_reason = 'process exited'
        while proc.poll() is None:
            time.sleep(args.poll_seconds)
            mon = load_monitor(out_path, name)
            latest, best = get_latest_best(mon)
            metrics = latest.get('metrics') or {}
            print(
                f'[{datetime.now()}] {name} batch={latest.get("batch")} '
                f'ndcg={latest.get("value")} hr10={metrics.get("hr_10")} '
                f'ads_hr10={metrics.get("ads_hr_10")} ads_ndcg10={metrics.get("ads_ndcg_10")} '
                f'best_ads={state.get("best_ads_hr10")}',
                flush=True,
            )
            stop, reason = should_stop(latest, state['started_at'], args.max_batches, 5, state)
            if stop:
                stop_reason = reason
                terminate(proc)
                break
        mon = load_monitor(out_path, name)
        latest, best = get_latest_best(mon)
        record = {
            'name': name,
            'sigma_seconds': sigma,
            'page_title_gate': page_gate,
            'stop_reason': stop_reason,
            'returncode': proc.poll(),
            'latest': latest,
            'best_by_ndcg': best,
            'best_ads_hr10_seen': state.get('best_ads_hr10'),
            'best_overall_hr10_seen': state.get('best_overall_hr10'),
            'log': str(log_path),
            'config': str(config_path),
            'out': str(out_path),
            'ended_at': datetime.now(timezone.utc).isoformat(),
        }
        with summary_path.open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(f'[{datetime.now()}] END {name}: {stop_reason}', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
