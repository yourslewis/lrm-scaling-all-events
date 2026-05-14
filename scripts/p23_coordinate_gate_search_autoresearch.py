#!/usr/bin/env python3
from __future__ import annotations
import json, os, signal, subprocess, sys, time, threading
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/home/yourslewis/lrm-scaling-all-events')
BASE=ROOT/'proposed_2-mmoe_ple/config/proposed23_coordinate_base_s300_page07.gin'
TRAIN=ROOT/'proposed_2-mmoe_ple/train'
DATA=Path('/home/yourslewis/lrm_benchmarkv4/processed/all_events_v2')
EMB=Path('/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings')
OUTROOT=ROOT/'results_v2'
RUNNER_DIR=OUTROOT/'p23_coordinate_gate_search_autoresearch'
CFGDIR=ROOT/'proposed_2-mmoe_ple/config/generated_p23_coordinate_search'
MAX_BATCH=26000
POLL=60
BASELINE={'strong':1.0,'page':0.7,'msn':0.1,'outlook':0.0}
BASELINE_METRICS={'overall_hr10':0.6056,'ads_hr10':0.2121,'overall_ndcg10':0.5590,'ads_ndcg10':0.1467,'score':0.4*0.6056+0.6*0.2121}
STAGES=[
    ('page', [0.5,0.6,0.8,0.9]),
    ('msn', [0.0,0.2,0.3,0.4]),
    ('strong', [0.7,0.8,0.9]),
    ('outlook', [0.1,0.2]),
]

def ensure_dirs():
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    CFGDIR.mkdir(parents=True, exist_ok=True)

def score(overall, ads):
    if overall is None or ads is None: return None
    return 0.4*float(overall)+0.6*float(ads)

def candidate_name(stage, gates):
    return f"p23_{stage}_s{int(gates['strong']*10):02d}_p{int(gates['page']*10):02d}_m{int(gates['msn']*10):02d}_o{int(gates['outlook']*10):02d}"

def patch_config(name, gates):
    text=BASE.read_text()
    text=text.replace('proposed23_coordinate_base_s300_page07', name)
    repl={
        'make_model.ad_anchor_strong_event_gate = 1.0': f"make_model.ad_anchor_strong_event_gate = {gates['strong']:.3f}",
        'make_model.ad_anchor_page_title_gate = 0.7': f"make_model.ad_anchor_page_title_gate = {gates['page']:.3f}",
        'make_model.ad_anchor_msn_gate = 0.1': f"make_model.ad_anchor_msn_gate = {gates['msn']:.3f}",
        'make_model.ad_anchor_outlook_gate = 0.0': f"make_model.ad_anchor_outlook_gate = {gates['outlook']:.3f}",
    }
    for old,new in repl.items():
        if old not in text: raise RuntimeError(f'missing config pattern: {old}')
        text=text.replace(old,new)
    return text

def launch(name,cfg,gpu,port):
    out=OUTROOT/name; logdir=out/'logs'; logdir.mkdir(parents=True, exist_ok=True)
    cfgp=CFGDIR/f'{name}.gin'; cfgp.write_text(cfg)
    log=logdir/f'train_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.log'
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu); env['AZUREML_RUN_ID']=f'{name}_{datetime.now(timezone.utc).strftime("%Y%m%d")}'; env['PYTHONPATH']=f'{TRAIN}:{env.get("PYTHONPATH","")}'
    cmd=['torchrun','--nproc_per_node=1',f'--master_port={port}','main.py',f'--gin_config_file={cfgp}',f'--output_path={out}','--data_path',str(DATA),'--mode=job','--ads_semantic_embd_path',str(EMB/'domain_0'),'--web_browsing_semantic_embd_path',str(EMB/'domain_1'),'--shopping_semantic_embd_path',str(EMB/'domain_2'),'--ads_pure_corpus_embd_path',str(EMB/'domain_3'),'--other_semantic_embd_path',str(EMB/'domain_4')]
    f=open(log,'w')
    p=subprocess.Popen(cmd,cwd=TRAIN,env=env,stdout=f,stderr=subprocess.STDOUT,preexec_fn=os.setsid)
    (out/'train.pid').write_text(str(p.pid))
    return p,out,log,cfgp

def monitor(out):
    files=sorted(out.glob('*/validation_monitor.json'), key=lambda p:p.stat().st_mtime, reverse=True)
    if not files: return None
    try: return json.loads(files[0].read_text())
    except Exception: return None

def metrics(entry):
    if not entry: return {}
    m=entry.get('metrics') or {}
    return {'batch':entry.get('batch'),'overall_ndcg10':entry.get('value') if entry.get('value') is not None else m.get('ndcg_10'),'overall_hr10':m.get('hr_10'),'ads_hr10':m.get('ads_hr_10'),'ads_ndcg10':m.get('ads_ndcg_10')}

def has_nan_metrics(entry):
    if not entry: return False
    m=entry.get('metrics') or {}
    for v in m.values():
        if isinstance(v,float) and v != v: return True
    val=entry.get('value')
    return isinstance(val,float) and val != val

def terminate(proc):
    if proc.poll() is not None: return
    try: os.killpg(proc.pid, signal.SIGTERM)
    except Exception: pass
    time.sleep(8)
    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except Exception: pass

def run_candidate(stage,gates,gpu,port):
    name=candidate_name(stage,gates)
    proc,out,log,cfgp=launch(name,patch_config(name,gates),gpu,port)
    print(f'START {name} stage={stage} gates={gates} gpu={gpu} pid={proc.pid}', flush=True)
    best_ads=0.0; best_overall=0.0; best_score=0.0; reason='process exited'
    while proc.poll() is None:
        time.sleep(POLL)
        mon=monitor(out); latest=(mon or {}).get('latest') or {}; lm=metrics(latest); b=lm.get('batch') or 0
        if has_nan_metrics(latest):
            reason='invalid NaN metrics'; terminate(proc); break
        ads=lm.get('ads_hr10'); overall=lm.get('overall_hr10'); sc=score(overall,ads)
        if ads is not None: best_ads=max(best_ads,float(ads))
        if overall is not None: best_overall=max(best_overall,float(overall))
        if sc is not None: best_score=max(best_score,float(sc))
        print(f'{name} batch={b} overall={overall} ads={ads} score={sc} best_ads={best_ads}', flush=True)
        if b >= MAX_BATCH:
            reason=f'reached {MAX_BATCH}'; terminate(proc); break
        if b >= 12000 and best_ads < 0.145:
            reason=f'early reject best_ads={best_ads:.4f}'; terminate(proc); break
        if b >= 20000 and best_ads < 0.1744:
            reason=f'reject below P18 best_ads={best_ads:.4f}'; terminate(proc); break
    mon=monitor(out); latest=metrics((mon or {}).get('latest')); best=metrics((mon or {}).get('best'))
    latest_score=score(latest.get('overall_hr10'), latest.get('ads_hr10'))
    rec={'name':name,'stage':stage,'gates':gates,'reason':reason,'returncode':proc.poll(),'latest':latest,'best_by_ndcg':best,'best_ads_hr10_seen':best_ads,'best_overall_hr10_seen':best_overall,'best_score_seen':best_score,'latest_score':latest_score,'passes_guardrail': bool(latest_score is not None and (latest.get('overall_hr10') or 0)>=0.58 and (latest.get('ads_hr10') or 0)>=0.2121),'log':str(log),'config':str(cfgp),'out':str(out),'ended_at':datetime.now(timezone.utc).isoformat()}
    with (RUNNER_DIR/'summary.jsonl').open('a') as f: f.write(json.dumps(rec)+'\n')
    print(f'END {name}: {reason}', flush=True)
    return rec

def run_wave(stage, candidates):
    results=[]; lock=threading.Lock()
    def worker(gpu, items):
        for idx,gates in enumerate(items):
            rec=run_candidate(stage,gates,gpu,30431+gpu+idx*10)
            with lock: results.append(rec)
    queues=[candidates[::2], candidates[1::2]]
    threads=[]
    for gpu,q in enumerate(queues):
        t=threading.Thread(target=worker,args=(gpu,q)); t.start(); threads.append(t)
    for t in threads: t.join()
    return results

def best_from_stage(current, stage_results):
    best={'gates':dict(current),'score':BASELINE_METRICS['score'],'source':'current'}
    # If current has been updated before, read state score.
    state_path=RUNNER_DIR/'state.json'
    if state_path.exists():
        st=json.loads(state_path.read_text()); best={'gates':st['current_gates'],'score':st['current_score'],'source':st.get('source','state')}
    for r in stage_results:
        latest=r.get('latest') or {}; sc=r.get('latest_score')
        if r.get('passes_guardrail') and sc is not None and sc > best['score']:
            best={'gates':r['gates'],'score':sc,'source':r['name']}
    state_path.write_text(json.dumps({'current_gates':best['gates'],'current_score':best['score'],'source':best['source'],'updated_at':datetime.now(timezone.utc).isoformat()},indent=2))
    return best

def main():
    ensure_dirs()
    current=dict(BASELINE)
    (RUNNER_DIR/'state.json').write_text(json.dumps({'current_gates':current,'current_score':BASELINE_METRICS['score'],'source':'p20_s300_page07','updated_at':datetime.now(timezone.utc).isoformat()},indent=2))
    for stage, values in STAGES:
        candidates=[]
        for v in values:
            g=dict(current); g[stage]=v; candidates.append(g)
        results=run_wave(stage,candidates)
        best=best_from_stage(current, results)
        current=dict(best['gates'])
        print(f'STAGE_DONE {stage} best={best}', flush=True)
    print('ALL_DONE', current, flush=True)

if __name__=='__main__': main()
