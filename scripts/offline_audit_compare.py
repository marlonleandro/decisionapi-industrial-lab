"""Append-only local audit, validation freeze and paired test. No API calls."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [c-h, c+h]


def metrics(rows, threshold):
    counts = Counter()
    for r in rows:
        y = r['ground_truth'] == 'anomaly'
        p = float(r['probability']) >= threshold
        counts['tp' if y and p else 'fn' if y else 'fp' if p else 'tn'] += 1
    tp, tn, fp, fn = (counts[k] for k in ('tp','tn','fp','fn'))
    n = len(rows)
    return dict(samples=n, threshold=threshold, tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=(tp+tn)/n, precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn), f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                fpr=fp/(fp+tn), fnr=fn/(fn+tp),
                accuracy_ci95_wilson=wilson(tp+tn,n),
                recall_ci95_wilson=wilson(tp,tp+fn), fpr_ci95_wilson=wilson(fp,fp+tn))


def checked_rows(root, manifest, path, split, basename=False):
    rows = read_csv(path)
    expected = {r['image']: r['ground_truth'] for r in manifest if r['split']==split}
    by_key = {}
    for r in rows:
        key = r['image'].replace('\\','/')
        if basename:
            matches = [k for k,v in expected.items() if Path(k).name==key and v==r['ground_truth']]
            require(len(matches)==1, 'Ambiguous YOLO image')
            key = matches[0]
        require(key in expected and r['ground_truth']==expected[key], 'Image or label mismatch')
        require(key not in by_key, 'Duplicate prediction')
        require(r.get('status','ok')=='ok' and not r.get('error',''), 'Failed prediction')
        require(math.isfinite(float(r['probability'])) and 0<=float(r['probability'])<=1, 'Invalid score')
        by_key[key] = {**r, 'image':key}
    require(set(by_key)==set(expected), 'Incomplete predictions')
    return [by_key[k] for k in sorted(expected)]


def audit(root, out):
    require(not out.exists(), 'Audit output already exists; use a new --output')
    manifest = read_csv(root/'data/pcb1_splits.csv')
    exploration = read_csv(root/'data/benchmark_pcb1.csv')
    require(len(manifest)==340 and len({r['image'] for r in manifest})==340, 'Invalid manifest')
    require(Counter((r['split'],r['ground_truth']) for r in manifest)==Counter({('train','normal'):200,('train','anomaly'):40,('validation','normal'):20,('validation','anomaly'):20,('test','normal'):30,('test','anomaly'):30}), 'Wrong counts')
    require(len(exploration)==20 and len({r['image'] for r in exploration})==20, 'Invalid exploration')
    require(not {r['image'] for r in exploration}&{r['image'] for r in manifest}, 'Exploration leakage')
    raw = root/'data/raw/pcb1/Data/Images'
    require(len(list((raw/'Normal').glob('*.JPG')))==1004 and len(list((raw/'Anomaly').glob('*.JPG')))==100, 'Raw counts differ')
    reproduced = []
    rng = random.Random(42)
    excluded = {(root/r['image']).resolve() for r in exploration}
    for label, folder in [('normal','Normal'),('anomaly','Anomaly')]:
        files = [p for p in sorted((raw/folder).iterdir()) if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'} and p.resolve() not in excluded]
        rng.shuffle(files)
        offset = 0
        for split, count in [('validation',20),('test',30),('train',200 if label=='normal' else 40)]:
            reproduced += [{'image':p.relative_to(root).as_posix(),'category':'pcb1','ground_truth':label,'split':split} for p in files[offset:offset+count]]
            offset += count
    rng.shuffle(reproduced)
    require(reproduced==manifest, 'Seed 42 does not reproduce manifest')
    hashes, copies = defaultdict(set), []
    for r in exploration+manifest:
        digest = sha(root/r['image'])
        hashes[digest].add(r.get('split','exploration'))
    require(all(len(v)==1 for v in hashes.values()), 'Cross-split byte duplicate')
    for split in ['train','validation','test']:
        expected = set()
        for r in manifest:
            if r['split']!=split: continue
            rel = Path('data/yolo_pcb1')/('val' if split=='validation' else split)/r['ground_truth']/Path(r['image']).name
            require(sha(root/rel)==sha(root/r['image']), 'YOLO copy differs')
            expected.add(rel.as_posix())
            copies.append({'source':r['image'],'copy':rel.as_posix(),'sha256':sha(root/rel)})
        actual = {p.relative_to(root).as_posix() for p in (root/'data/yolo_pcb1'/('val' if split=='validation' else split)).rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}}
        require(actual==expected, 'Extra/missing YOLO files')
    api = {}
    for split in ['validation','test']:
        rows = checked_rows(root,manifest,root/f'results/runs/v1_{split}/predictions.csv',split)
        for r in rows:
            key = hashlib.sha256(r['image'].encode()).hexdigest()[:16]
            c = json.loads((root/f'results/runs/v1_{split}/checkpoints/{key}.json').read_text())
            require(all(c[k]==r[k] for k in ['image','ground_truth','split','prompt_version','model','status']), 'Checkpoint metadata mismatch')
            require(float(c['probability'])==float(r['probability'])==float(c['raw_response']['answers'][0]['probability']), 'Checkpoint score mismatch')
        api[split] = metrics(rows,0.2)
    selected = json.loads((root/'results/selected_threshold.json').read_text())
    require(selected['selection']['threshold']==0.2, 'API threshold changed')
    for k,v in selected['selection'].items():
        require(abs(api['validation'][k]-v)<1e-12, 'API validation metric mismatch')
    final = json.loads((root/'results/final_test_metrics.json').read_text())
    for k in ['samples','accuracy','precision','recall','f1','fpr','fnr','tp','tn','fp','fn','threshold']:
        require(abs(api['test'][k]-final[k])<1e-12, 'API test metric mismatch')
    historical_yolo = checked_rows(root,manifest,root/'results/yolo/pcb1_cls_v1/predictions_val.csv','validation',True)
    smoke = json.loads((root/'data/samples/samples.json').read_text())['samples']
    smoke_overlap = [{**r,'split':next((m['split'] for m in manifest if m['image']==r['source']),None)} for r in smoke]
    snapshot = {}
    for base in ['scripts','src','docs','results','weights']:
        for p in (root/base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                snapshot[p.relative_to(root).as_posix()] = sha(p)
    for name in ['README.md','requirements.txt','.gitignore','data/pcb1_splits.csv','data/benchmark_pcb1.csv','yolo11n-cls.pt']:
        snapshot[name] = sha(root/name)
    out.mkdir(parents=True)
    write_json(out/'original_hashes.json',snapshot)
    write_csv(out/'dataset_hashes.csv',copies)
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),checks_passed=True,
                  manifest_sha256=sha(root/'data/pcb1_splits.csv'),model_sha256=sha(root/'results/yolo/pcb1_cls_v1/weights/best.pt'),
                  split_counts={f'{s}/{l}':n for (s,l),n in Counter((r['split'],r['ground_truth']) for r in manifest).items()},
                  seed42_reproduced=True, cross_split_byte_duplicates=0, yolo_copies_verified=len(copies),
                  initial_smoke_overlap=smoke_overlap, api_metrics=api,
                  historical_yolo_val=metrics(historical_yolo,0.5),
                  caveat='Initial smoke samples were not excluded by the split generator. Test independence is assessed separately from this validation exposure.')
    write_json(out/'audit.json',result)
    print(json.dumps(result,indent=2))


def infer(root, manifest, split, modelpath):
    import torch
    from ultralytics import YOLO
    require(torch.cuda.is_available(), 'CUDA unavailable; explicit review required')
    model = YOLO(str(modelpath))
    require(set(model.names.values())=={'normal','anomaly'}, 'Wrong model classes')
    aid = next(k for k,v in model.names.items() if v=='anomaly')
    rows=[]
    for r in sorted((r for r in manifest if r['split']==split),key=lambda r:r['image']):
        torch.cuda.synchronize()
        start=time.perf_counter()
        pred=model.predict(str(root/r['image']),imgsz=320,device=0,verbose=False,save=False)[0]
        torch.cuda.synchronize()
        rows.append({**r,'probability':float(pred.probs.data[aid].item()),'latency_ms':(time.perf_counter()-start)*1000})
    return rows,dict(python=sys.version,torch=torch.__version__,cuda_runtime=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0),ultralytics=__import__('ultralytics').__version__,
                     transforms=str(model.predictor.transforms),names=model.names)


def freeze(root,out):
    audited=json.loads((out/'audit.json').read_text())
    require(audited['checks_passed'], 'Audit failed')
    require(not (out/'yolo_threshold_frozen.json').exists(), 'Threshold already frozen')
    require(sha(root/'data/pcb1_splits.csv')==audited['manifest_sha256'], 'Manifest changed')
    modelpath=root/'results/yolo/pcb1_cls_v1/weights/best.pt'
    require(sha(modelpath)==audited['model_sha256'], 'Model changed')
    manifest=read_csv(root/'data/pcb1_splits.csv')
    rows,env=infer(root,manifest,'validation',modelpath)
    historical=checked_rows(root,manifest,root/'results/yolo/pcb1_cls_v1/predictions_val.csv','validation',True)
    maxdelta=max(abs(float(a['probability'])-float(b['probability'])) for a,b in zip(rows,historical))
    require(maxdelta<1e-5, 'Current model does not reproduce historical validation scores')
    candidates=[metrics(rows,i/100) for i in range(5,96,5)]
    feasible=[m for m in candidates if m['fpr']<=0.10]
    require(bool(feasible), 'No feasible threshold')
    # Same policy and ascending-grid first tie as existing API freeze script.
    selected=max(feasible,key=lambda m:(m['recall'],m['f1'],-m['fpr']))
    write_csv(out/'yolo_predictions_validation.csv',rows)
    write_json(out/'yolo_threshold_grid_validation.json',candidates)
    write_json(out/'environment.json',env)
    frozen=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),model_sha256=sha(modelpath),
                manifest_sha256=sha(root/'data/pcb1_splits.csv'),validation_predictions_sha256=sha(out/'yolo_predictions_validation.csv'),
                policy='FPR <= 0.10; maximize recall, then F1, then minimize FPR; ascending grid first tie',
                grid=[i/100 for i in range(5,96,5)],selection=selected, historical_val_max_probability_delta=maxdelta)
    write_json(out/'yolo_threshold_frozen.json',frozen)
    print(json.dumps(frozen,indent=2))


def test(root,out):
    frozen=json.loads((out/'yolo_threshold_frozen.json').read_text())
    require(not (out/'yolo_predictions_test.csv').exists(), 'Test already evaluated; reuse saved results')
    require(sha(root/'data/pcb1_splits.csv')==frozen['manifest_sha256'], 'Manifest changed')
    modelpath=root/'results/yolo/pcb1_cls_v1/weights/best.pt'
    require(sha(modelpath)==frozen['model_sha256'], 'Model changed')
    require(sha(out/'yolo_predictions_validation.csv')==frozen['validation_predictions_sha256'], 'Validation changed')
    manifest=read_csv(root/'data/pcb1_splits.csv')
    # Recheck image identities before inference, without changing partitions.
    for r in read_csv(out/'dataset_hashes.csv'):
        require(sha(root/r['source'])==r['sha256']==sha(root/r['copy']), 'Dataset changed')
    rows,env=infer(root,manifest,'test',modelpath)
    threshold=frozen['selection']['threshold']
    ym=metrics(rows,threshold)
    ar=checked_rows(root,manifest,root/'results/runs/v1_test/predictions.csv','test')
    am=metrics(ar,0.2)
    paired=[]
    for y,a in zip(rows,ar):
        require(y['image']==a['image'], 'Pairing mismatch')
        truth=y['ground_truth']=='anomaly'
        yp=float(y['probability'])>=threshold
        ap=float(a['probability'])>=0.2
        paired.append(dict(image=y['image'],ground_truth=y['ground_truth'],yolo_probability=y['probability'],api_probability=float(a['probability']),yolo_prediction='anomaly' if yp else 'normal',api_prediction='anomaly' if ap else 'normal',yolo_correct=yp==truth,api_correct=ap==truth))
    b=sum(r['yolo_correct'] and not r['api_correct'] for r in paired)
    c=sum(r['api_correct'] and not r['yolo_correct'] for r in paired)
    n=b+c
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),test_independent_of_training=True,
                yolo=ym,decisions_api=am,paired_discordance=dict(yolo_only_correct=b,api_only_correct=c,mcnemar_exact_two_sided_p=p),
                warning='Small balanced local split; test was previously inspected for API, YOLO threshold selected on validation only. No production or causal superiority claim.',
                latency=dict(yolo_mean_ms=sum(r['latency_ms'] for r in rows)/len(rows),yolo_excluding_first_mean_ms=sum(r['latency_ms'] for r in rows[1:])/len(rows[1:]),api_historical_mean_ms=sum(float(r['latency_ms']) for r in ar)/len(ar),comparable=False))
    write_csv(out/'yolo_predictions_test.csv',rows)
    write_csv(out/'paired_test.csv',paired)
    write_json(out/'comparison_test.json',result)
    write_json(out/'environment_test.json',env)
    print(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['audit','freeze','test'])
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output',default='results/audit_20261013')
    args=ap.parse_args()
    root=args.root.resolve()
    out=(root/args.output).resolve()
    require(out.is_relative_to(root/'results'), 'Output must be inside repository results')
    {'audit':audit,'freeze':freeze,'test':test}[args.stage](root,out)


if __name__=='__main__':
    main()
