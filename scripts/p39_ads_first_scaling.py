#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, signal, subprocess, time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT=Path(os.environ.get('LRM_ROOT','/home/yourslewis/lrm-scaling-all-events'))
TRAIN=ROOT/'proposed_2-mmoe_ple/train'
OUTROOT=ROOT/'results_v2/p39_ads_first_scaling'
DATA_FULL=Path('/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve')
DATA_FROZEN=Path('/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_eval7raw_0_6')
EMB=Path('/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve')
PYTHON=os.environ.get('LRM_PYTHON','/home/yourslewis/miniconda3/envs/hstu/bin/python3.10')

@dataclass(frozen=True)
class Profile:
    id:str; config:str; scale:float; summary:str

PROFILES={p.id:p for p in [
    Profile('p39a_full','proposed_2-mmoe_ple/config/generated_p39_ads_first_scaling/p39a_p29b_ads_only_full.gin',1.0,'P29B Ads-only supervision, full v3 train'),
    Profile('p39b_20','proposed_2-mmoe_ple/config/generated_p39_ads_first_scaling/p39b_p29b_ads_aux_light.gin',0.2,'P29B Ads-first light auxiliary, 20% train'),
    Profile('p39b_50','proposed_2-mmoe_ple/config/generated_p39_ads_first_scaling/p39b_p29b_ads_aux_light.gin',0.5,'P29B Ads-first light auxiliary, 50% train'),
    Profile('p39b_full','proposed_2-mmoe_ple/config/generated_p39_ads_first_scaling/p39b_p29b_ads_aux_light.gin',1.0,'P29B Ads-first light auxiliary, full v3 train'),
    Profile('p39c_full','proposed_2-mmoe_ple/config/generated_p39_ads_first_scaling/p39c_p29b_ads_aux_medium.gin',1.0,'P29B Ads-first medium auxiliary, full v3 train'),
]}

def utc(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')

def data_dir(scale:float)->Path:
    name='full' if scale>=0.999 else f'{int(scale*100):02d}pct'
    d=Path(f'/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve_train_{name}_frozen7eval')
    d.mkdir(parents=True,exist_ok=True)
    train=d/'train'; train.mkdir(exist_ok=True)
    parts=sorted((DATA_FULL/'train').glob('*.parquet'))
    n=len(parts) if scale>=0.999 else max(1, int(round(len(parts)*scale)))
    keep=parts[:n]
    for p in keep:
        link=train/p.name
        if not link.exists(): link.symlink_to(p)
    for stale in train.glob('*.parquet'):
        if stale.name not in {p.name for p in keep}: stale.unlink()
    for name,target in [('eval',DATA_FROZEN/'eval'),('metadata.json',DATA_FROZEN/'metadata.json'),('freeze_manifest.json',DATA_FROZEN/'freeze_manifest.json')]:
        path=d/name
        if not path.exists(): path.symlink_to(target, target_is_directory=target.is_dir())
    (d/'DATASET_SCALE.json').write_text(json.dumps({'scale':scale,'train_parts':n,'source_train':str(DATA_FULL/'train'),'frozen_eval':str(DATA_FROZEN/'eval'),'updated_at':utc()},indent=2))
    return d

def env(gpu:int,port:int,run_id:str):
    e=os.environ.copy(); e['CUDA_VISIBLE_DEVICES']=str(gpu); e['AZUREML_RUN_ID']=run_id; e['PYTHONPATH']=f'{TRAIN}:{e.get("PYTHONPATH","")}'
    e.update({'TOKENIZERS_PARALLELISM':'false','TORCHDYNAMO_DISABLE':'1','TORCHINDUCTOR_COMPILE_THREADS':'1','RANK':'0','WORLD_SIZE':'1','LOCAL_RANK':'0','MASTER_ADDR':'127.0.0.1','MASTER_PORT':str(port)})
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: e[k]='1'
    return e

def launch(p:Profile,gpu:int,port:int):
    cfg=ROOT/p.config; dd=data_dir(p.scale); out=OUTROOT/p.id; logdir=out/'logs'; logdir.mkdir(parents=True,exist_ok=True)
    log=logdir/f'train_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.log'
    cmd=[PYTHON,'-u','main.py',f'--gin_config_file={cfg}',f'--output_path={out}','--data_path',str(dd),'--mode=job','--ads_semantic_embd_path',str(EMB/'domain_0'),'--web_browsing_semantic_embd_path',str(EMB/'domain_1'),'--shopping_semantic_embd_path',str(EMB/'domain_2'),'--ads_pure_corpus_embd_path',str(EMB/'domain_3'),'--other_semantic_embd_path',str(EMB/'domain_4')]
    f=open(log,'w'); proc=subprocess.Popen(cmd,cwd=TRAIN,env=env(gpu,port,f'{p.id}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'),stdout=f,stderr=subprocess.STDOUT,preexec_fn=os.setsid)
    (out/'launch.json').write_text(json.dumps({'profile':asdict(p),'cmd':cmd,'data_path':str(dd),'started_at':utc()},indent=2)); return proc,out,log,cfg,dd

def terminate(proc):
    if proc.poll() is not None: return
    try: os.killpg(proc.pid, signal.SIGTERM)
    except Exception: pass
    time.sleep(20)
    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except Exception: pass

def monitor(out:Path):
    files=sorted(out.glob('*/validation_monitor.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not files: return None,{}
    try: return files[0],json.loads(files[0].read_text())
    except Exception: return files[0],{}

def log_batch(log:Path):
    if not log.exists(): return 0
    m=re.findall(r'batch-stat \(train\): iteration (\d+)', log.read_text(errors='ignore')[-200000:])
    return int(m[-1]) if m else 0

def score(entry):
    m=(entry or {}).get('metrics') or {}; v=m.get('ads_hr_10')
    return None if v is None else float(v)

def metrics(entry):
    m=(entry or {}).get('metrics') or {}; return {'batch':(entry or {}).get('batch'),'AHR':m.get('ads_hr_10'),'OHR':m.get('hr_10'),'A_NDCG':m.get('ads_ndcg_10'),'O_NDCG':m.get('ndcg_10'),'score':score(entry or {})}

def monitor_run(proc,out,log,min_batch,patience,max_batch,poll):
    best_s=None; best_b=0; reason='process_exited'
    while proc.poll() is None:
        time.sleep(poll); mp,mon=monitor(out); latest=mon.get('latest') or {}; b=int(latest.get('batch') or log_batch(log) or 0); s=score(latest)
        if s is not None and (best_s is None or s>best_s+1e-9): best_s=s; best_b=b
        print(json.dumps({'time':utc(),'profile':out.name,'batch':b,'AHR':s,'best_AHR':best_s,'best_batch':best_b,'monitor':str(mp) if mp else None}),flush=True)
        if b>=max_batch: reason=f'reached_max_batch_{max_batch}'; terminate(proc); break
        if b>=min_batch and best_b and b-best_b>=patience: reason=f'early_stop_AHR_peaked_after_{patience}_batches'; terminate(proc); break
    mp,mon=monitor(out); latest=mon.get('latest') or {}; return {'reason':reason,'returncode':proc.poll(),'monitor':str(mp) if mp else None,'latest':metrics(latest),'best_AHR_batch':best_b or None,'best_AHR':best_s,'log':str(log)}

def best_ckpt(out,stop):
    b=stop.get('best_AHR_batch')
    if b is not None:
        c=sorted(out.glob(f'*/ckpts/checkpoint_batch{int(b):07d}.pt'),key=lambda p:p.stat().st_mtime,reverse=True)
        if c: return c[0]
    c=sorted(out.glob('*/ckpts/best_checkpoint_*.pt'),key=lambda p:p.stat().st_mtime,reverse=True)
    return c[0] if c else None

def final_eval(p,cfg,ckpt,dd,gpu,port):
    out=OUTROOT/p.id; outj=out/f'{p.id}_frozen7_eval.json'; log=out/'logs'/f'eval_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.log'
    cmd=['torchrun','--nproc_per_node=1',f'--master_port={port}','evaluate_per_domain.py',f'--gin_config_file={cfg}',f'--checkpoint_path={ckpt}',f'--data_path={dd}',f'--ads_semantic_embd_path={EMB/"domain_0"}',f'--web_browsing_semantic_embd_path={EMB/"domain_1"}',f'--shopping_semantic_embd_path={EMB/"domain_2"}',f'--ads_pure_corpus_embd_path={EMB/"domain_3"}',f'--other_semantic_embd_path={EMB/"domain_4"}','--max_eval_batches=1000000','--eval_batch_size=16','--mode=job',f'--output_json={outj}']
    with open(log,'w') as f: rc=subprocess.call(cmd,cwd=TRAIN,env=env(gpu,port,f'eval_{p.id}'),stdout=f,stderr=subprocess.STDOUT)
    res={'returncode':rc,'output_json':str(outj),'log':str(log)}
    if outj.exists():
        raw=json.loads(outj.read_text()); o=raw.get('overall') or {}; ads=(raw.get('per_domain') or {}).get('Ads') or {}
        res.update({'OHR':o.get('hr_10'),'AHR':ads.get('hr_10',o.get('ads_hr_10')),'O_NDCG':o.get('ndcg_10'),'A_NDCG':ads.get('ndcg_10',o.get('ads_ndcg_10'))})
    return res

def write_metrics(records):
    rows=[]
    for r in records:
        ev=r.get('final_eval') or {}; rows.append({'id':r['id'],'label':r['summary'],'OHR':ev.get('OHR'),'AHR':ev.get('AHR'),'O_NDCG':ev.get('O_NDCG'),'A_NDCG':ev.get('A_NDCG'),'source':ev.get('output_json'),'note':r.get('stop',{}).get('reason')})
    (OUTROOT/'p39_ads_first_scaling_metrics.json').write_text(json.dumps({'updated_at_utc':utc(),'target_AHR':0.3066437,'rows':rows},indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--profiles',nargs='*',default=['p39a_full','p39b_20','p39b_50','p39b_full','p39c_full']); ap.add_argument('--gpu',type=int,default=0); ap.add_argument('--port',type=int,default=29940); ap.add_argument('--poll-seconds',type=int,default=300); ap.add_argument('--min-batch',type=int,default=12000); ap.add_argument('--patience-batches',type=int,default=8000); ap.add_argument('--max-batch',type=int,default=70000); args=ap.parse_args()
    OUTROOT.mkdir(parents=True,exist_ok=True); state=OUTROOT/'state.json'; records=[]
    if state.exists():
        try: records=json.loads(state.read_text()).get('records',[])
        except Exception: records=[]
    done={r.get('id') for r in records if r.get('status')=='done'}
    for i,pid in enumerate(args.profiles):
        if pid in done: continue
        p=PROFILES[pid]; proc,out,log,cfg,dd=launch(p,args.gpu,args.port+i*2); stop=monitor_run(proc,out,log,args.min_batch,args.patience_batches,args.max_batch,args.poll_seconds); ckpt=best_ckpt(out,stop); ev=final_eval(p,cfg,ckpt,dd,args.gpu,args.port+i*2+1) if ckpt else {'error':'no_checkpoint'}
        rec={'id':p.id,'summary':p.summary,'profile':asdict(p),'checkpoint':str(ckpt) if ckpt else None,'stop':stop,'final_eval':ev,'status':'done','updated_at_utc':utc()}; records.append(rec); state.write_text(json.dumps({'updated_at_utc':utc(),'records':records},indent=2)); write_metrics(records)
if __name__=='__main__': main()
