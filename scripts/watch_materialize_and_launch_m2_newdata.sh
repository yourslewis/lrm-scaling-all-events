#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yourslewis/lrm-launches/m2-newdata-src-20260530
DST=/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_canonical_row_array_v001_hstu_seqview
SRC=/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_canonical_row_array_v001
LOG=$ROOT/results_v2/m2_newdata_baseline/watch_materialize_and_launch.log
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start"
while pgrep -u yourslewis -f /tmp/materialize_v001_hstu_view_parallel.py >/dev/null; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] materializer still running: $(cat "$DST/MATERIALIZE_PROGRESS.json" 2>/dev/null || true)"
  sleep 60
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] materializer exited"
PY=/home/yourslewis/miniconda3/envs/hstu/bin/python
$PY - <<'PY'
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json, pyarrow.parquet as pq, shutil
SRC=Path('/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_canonical_row_array_v001')
OUT=Path('/home/yourslewis/lrm_benchmarkv4/processed/lrm_benchmark_v001_canonical_row_array_v001_hstu_seqview')
OLD_META=Path('/home/yourslewis/lrm_benchmarkv4/processed/all_events_v3_full_preserve/metadata.json')
DOMAIN_NAMES={0:'Ads',1:'Browsing',2:'Search',3:'Purchase',4:'Others'}
EVENT_TYPE_INFO={
 'NativeClick':(1,0,'Ads'),'SearchClick':(2,0,'Ads'),'EdgePageTitle':(3,1,'Browsing'),'EdgeSearchQuery':(4,2,'Search'),
 'OrganicSearchQuery':(5,2,'Search'),'UET':(6,1,'Browsing'),'OutlookSenderDomain':(7,4,'Others'),'UETShoppingCart':(8,3,'Purchase'),
 'UETShoppingView':(9,1,'Browsing'),'AbandonCart':(10,3,'Purchase'),'EdgeShoppingCart':(11,3,'Purchase'),'EdgeShoppingPurchase':(12,3,'Purchase'),
 'ChromePageTitle':(13,1,'Browsing'),'MSN':(14,1,'Browsing')}

def utc(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
def write_json(path,data):
    p=Path(path); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2,sort_keys=True),encoding='utf-8'); tmp.replace(p)
def cjson(c): return {str(k): int(v) for k,v in sorted(c.items(), key=lambda kv: str(kv[0]))}
def summarize_split(split):
    src_files=sorted((SRC/split).glob('part_*.parquet'))
    out_files=sorted((OUT/split).glob('part_*.parquet'))
    done_files=sorted((OUT/split).glob('part_*.parquet.done'))
    if len(out_files)!=len(src_files) or len(done_files)!=len(src_files):
        raise SystemExit(f'{split} incomplete: source={len(src_files)} parquet={len(out_files)} done={len(done_files)}')
    rows=events=targets=0; de=Counter(); te=Counter(); dt=Counter(); tt=Counter(); mn=mx=None
    for f in out_files:
        pf=pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=512, columns=['encoded_ids','types','timestamps_unix']):
            for r in batch.to_pylist():
                ids=r['encoded_ids'] or []; tys=r['types'] or []; tss=r['timestamps_unix'] or []
                rows+=1; events+=len(ids)
                if tss:
                    a=min(tss); b=max(tss); mn=a if mn is None else min(mn,a); mx=b if mx is None else max(mx,b)
                start=max(0, len(ids)-201)
                for i,ty in enumerate(tys):
                    dom=EVENT_TYPE_INFO.get(str(ty),(None,None,None))[1]
                    if dom is not None: de[dom]+=1
                    te[str(ty)]+=1
                    if i>=start+1:
                        targets+=1
                        if dom is not None: dt[dom]+=1
                        tt[str(ty)]+=1
    return {'rows':rows,'events':events,'parts':len(out_files),'effective_supervision_targets_maxseq200_all_position':targets,'domain_event_counts':cjson(de),'event_type_counts':cjson(te),'domain_target_counts_maxseq200_all_position':cjson(dt),'event_type_target_counts_maxseq200_all_position':cjson(tt),'min_event_time_unix_s':mn,'max_event_time_unix_s':mx}
source_summary=json.loads((SRC/'reports/generation_summary.json').read_text())
validation=json.loads((SRC/'reports/post_generation_validation_result.json').read_text())
summaries={split:summarize_split(split) for split in ['train','eval']}
if OLD_META.exists(): shutil.copy2(OLD_META, OUT/'metadata.json')
disclosure={
 'training_data_name':'lrm_benchmark_v001_canonical_row_array_v001_hstu_seqview',
 'training_data_path':str(OUT),
 'source_canonical_path':str(SRC),
 'dataset_version':source_summary.get('dataset_version'),
 'canonical_data_checksum':source_summary.get('canonical_data_checksum'),
 'generator_schema_version':source_summary.get('generator_schema_version'),
 'projection_policy':'all-event chronological sequence view; not Ads-projected',
 'event_time_window_utc':'2026-01-17T00:00:00Z to 2026-03-13T23:59:59Z (validated)',
 'domain_names':{str(k):v for k,v in DOMAIN_NAMES.items()},
 'event_type_info':{k:{'event_type_id':v[0],'domain_id':v[1],'domain':v[2]} for k,v in EVENT_TYPE_INFO.items()},
 'splits':summaries,
 'leakage_safety_notes':['Canonical post-generation validation passed before launch.' if validation.get('status')=='passed' else f"Canonical post-generation validation status: {validation.get('status')}", f"Train/eval user overlap: {validation.get('overlap_users')}.", 'Validation enforces chronological order, valid timestamps, no duplicate event_ids per row, no private/raw text-like schema fields, eval source lineage under all_events_v3_full_preserve_eval7raw_0_6/eval, and all events inside the fixed 2026-01-17..2026-03-13 window.', 'This HSTU seqview is a format transform only: canonical events -> encoded_ids/types/timestamps_unix, preserving chronological order.'],
 'embedding_item_universe':'/home/yourslewis/lrm_benchmarkv4/processed/semantic_embeddings_v3_full_preserve/domain_0..domain_4; metadata copied from all_events_v3_full_preserve/metadata.json',
 'transforms_filters':['Canonical generator excluded/quarantined out-of-window or invalid timestamp events; see source reports train/eval excluded_or_quarantined jsonl.','HSTU view drops canonical provenance/checksum columns from training parquet rows but keeps source disclosure and metadata sidecars.','HSTU loader uses max_sequence_length=200 and supervision_target_position=all over shifted labels.'],
 'created_at':utc(),
 'source_validation':validation,
}
write_json(OUT/'DATASET_DISCLOSURE.json', disclosure)
write_json(OUT/'DATASET_SCALE.json', {'scale':1.0,'source_canonical':str(SRC),'source_view':str(OUT),'created_at':utc()})
write_json(OUT/'MATERIALIZE_PROGRESS.json', {'status':'passed','finished_at':utc(),'source':str(SRC),'output':str(OUT),'summaries':summaries})
print(json.dumps({'status':'disclosure_written','train_rows':summaries['train']['rows'],'train_events':summaries['train']['events'],'train_targets':summaries['train']['effective_supervision_targets_maxseq200_all_position']}, indent=2))
PY

if pgrep -u yourslewis -f "m2_aux_light_hstu_newdata|results_v2/m2_newdata_baseline" >/dev/null; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] M2-newdata training already running"
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launching M2-newdata training"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES=0 LRM_ROOT="$ROOT" /home/yourslewis/miniconda3/envs/hstu/bin/python scripts/launch_m2_newdata.py
fi
sleep 5
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] post-launch procs"
ps -u yourslewis -o pid,ppid,etime,stat,%cpu,%mem,rss,cmd | grep -E "m2_aux_light_hstu_newdata|train/main.py|main.py" | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
