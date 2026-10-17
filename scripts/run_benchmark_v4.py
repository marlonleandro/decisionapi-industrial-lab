from __future__ import annotations
import argparse, csv, json, os, sys, time, hashlib
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.decisions.client import inspect_pcb_visible_defect

def atomic_json(path:Path, content:dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(content, ensure_ascii=False, indent=2),encoding='utf-8')
    os.replace(tmp,path)

def load_manifest(split):
    with (ROOT/'data/pcb1_splits.csv').open(newline='',encoding='utf-8') as f:
        records=[r for r in csv.DictReader(f) if r['split']==split]
    if not records: raise ValueError(f'Split vacio: {split}')
    return sorted(records,key=lambda r:r['image'])

def materialize_csv(run_dir:Path, manifest:list[dict]):
    out=[]
    for r in manifest:
        key=hashlib.sha256(r['image'].encode()).hexdigest()[:16]
        path=run_dir/'checkpoints'/f'{key}.json'
        if not path.is_file(): continue
        doc=json.loads(path.read_text(encoding='utf-8'))
        if doc['image'] != r['image']: raise ValueError(f'Hash conflict {path}')
        out.append({k:doc.get(k,'') for k in FIELDS})
    dst=run_dir/'predictions.csv'
    with dst.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(out)
    return dst,len(out)

FIELDS=['image','ground_truth','split','prompt_version','model','probability','latency_ms','status','error','timestamp_utc']

def main():
    ap=argparse.ArgumentParser(description='Benchmark con checkpoints/resume')
    ap.add_argument('--split',choices=['validation','test'],required=True)
    ap.add_argument('--prompt-version',default='v1',choices=['v1','v2'])
    ap.add_argument('--allow-test',action='store_true',help='Solo cuando el umbral haya sido congelado')
    ap.add_argument('--max-images',type=int,default=None,help='Para smoke test; corrida completa sin esta opcion')
    args=ap.parse_args()
    if args.split=='test' and not args.allow_test:
        ap.error('Test bloqueado; usar --allow-test solo despues de congelar el umbral de validacion.')
    if args.split=='test' and not (ROOT/'results/selected_threshold.json').is_file():
        ap.error('Falta results/selected_threshold.json. Primero calibra en validation.')
    load_dotenv(ROOT/'.env')
    manifest=load_manifest(args.split)
    if args.max_images is not None: manifest=manifest[:args.max_images]
    run_dir=ROOT/'results'/'runs'/f'{args.prompt_version}_{args.split}'
    (run_dir/'checkpoints').mkdir(parents=True,exist_ok=True)
    print(f'{args.prompt_version} {args.split}: {len(manifest)} imágenes')
    for i,r in enumerate(manifest,1):
        key=hashlib.sha256(r['image'].encode()).hexdigest()[:16]
        path=run_dir/'checkpoints'/f'{key}.json'
        if path.is_file():
            old=json.loads(path.read_text(encoding='utf-8'))
            if old.get('status')=='ok':
                print(f'[{i}/{len(manifest)}] cached {r["image"]}')
                continue
        image=ROOT/r['image']
        start=time.perf_counter()
        try:
            # El cliente original fue modificado en Sprint 3 para aceptar prompt_version.
            result=inspect_pcb_visible_defect(image,prompt_version=args.prompt_version)
            answer=next(a for a in result.answers if getattr(a,'name',None)=='visible_defect')
            prob=float(answer.probability)
            if not 0<=prob<=1: raise ValueError(f'Probabilidad fuera de rango: {prob}')
            doc={**r,'prompt_version':args.prompt_version,'model':getattr(result,'model','gpt-6-luna'),
                 'probability':prob,'latency_ms':round((time.perf_counter()-start)*1000,2),
                 'status':'ok','error':'','timestamp_utc':datetime.now(timezone.utc).isoformat(),
                 'raw_response':result.model_dump(mode='json')}
            atomic_json(path,doc)
            print(f'[{i}/{len(manifest)}] {image.name}  p={prob:.3f}')
        except Exception as exc:
            doc={**r,'prompt_version':args.prompt_version,'model':'gpt-6-luna',
                 'probability':None,'latency_ms':round((time.perf_counter()-start)*1000,2),
                 'status':'error','error':str(exc),'timestamp_utc':datetime.now(timezone.utc).isoformat()}
            atomic_json(path,doc)
            print(f'[{i}/{len(manifest)}] ERROR {image.name}: {exc}')
        dest,n=materialize_csv(run_dir,manifest)
    dest,n=materialize_csv(run_dir,manifest)
    print(f'\nCSV: {dest}\nRegistradas: {n}/{len(manifest)}')
    print('Errores se reintentan al volver a ejecutar; respuestas exitosas no se repiten.')

if __name__=='__main__': main()
