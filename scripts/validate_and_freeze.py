from __future__ import annotations
import csv,json
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score
ROOT=Path(__file__).resolve().parents[1]
CSV=ROOT/'results/runs/v1_validation/predictions.csv'
with CSV.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
valid=[r for r in rows if r['status']=='ok' and r['probability']!='']
if len(valid)!=40:
    raise SystemExit(f'Validacion incompleta: {len(valid)}/40 respuestas exitosas. Completa antes de congelar.')
y=[1 if r['ground_truth']=='anomaly' else 0 for r in valid]
p=[float(r['probability']) for r in valid]
# Politica ilustrativa: exigir FPR <= 10%, luego maximizar Recall, despues F1.
options=[]
for t in [i/100 for i in range(5,96,5)]:
    pred=[int(v>=t) for v in p]
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    fpr=fp/(fp+tn) if fp+tn else 0
    recall=tp/(tp+fn) if tp+fn else 0
    options.append(dict(threshold=round(t,2),fpr=fpr,recall=recall,f1=f1_score(y,pred,zero_division=0),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp)))
feasible=[r for r in options if r['fpr']<=0.10]
if not feasible:
    raise SystemExit('Ningun umbral cumple FPR <= 10%; revisar los criterios antes de congelar.')
selected=max(feasible,key=lambda r:(r['recall'],r['f1'],-r['fpr']))
out=ROOT/'results/selected_threshold.json'
out.parent.mkdir(exist_ok=True,parents=True)
if out.exists():
    raise SystemExit(f'{out} ya existe. No sobrescribir sin revision metodologica.')
out.write_text(json.dumps({'source':'v1_validation','selection_policy':'maximize recall subject to FPR <= 0.10; tie-break F1','selection':selected,'sample_count':len(valid)},indent=2),encoding='utf-8')
print('Umbral congelado (solo validación):')
print(json.dumps(selected,indent=2))
print('Archivo:',out)
