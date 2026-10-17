from __future__ import annotations
import argparse, csv, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALID_EXT = {'.jpg', '.jpeg', '.png', '.webp'}


def find_pcb1():
    for p in sorted((ROOT / 'data' / 'raw').rglob('pcb1')):
        if (p / 'Data' / 'Images' / 'Normal').is_dir() and (p / 'Data' / 'Images' / 'Anomaly').is_dir():
            return p
    raise FileNotFoundError('No se encontro pcb1 en data/raw; descarga VisA primero.')


def read_existing(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Falta {path}; es imprescindible excluir las 20 imágenes del Sprint 2.')
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 20:
        raise ValueError(f'Se esperaban 20 imágenes de exploracion, hay {len(rows)}.')
    paths = [(ROOT / r['image']).resolve() for r in rows]
    if len(set(paths)) != len(paths):
        raise ValueError('Hay imágenes duplicadas en benchmark_pcb1.csv.')
    return set(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    pcb = find_pcb1()
    excluded = read_existing(ROOT / 'data' / 'benchmark_pcb1.csv')
    rng = random.Random(args.seed)
    grouped = {}
    for label, name in [('normal','Normal'), ('anomaly','Anomaly')]:
        folder = pcb / 'Data' / 'Images' / name
        allfiles = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT)
        eligible = [p for p in allfiles if p.resolve() not in excluded]
        rng.shuffle(eligible)
        grouped[label] = eligible
        print(f'{label}: total={len(allfiles)}, previously_used={len(allfiles)-len(eligible)}, available={len(eligible)}')

    # 20 normal + 20 anomaly validation; 30 + 30 test; 200 normal + 40 anomaly training
    counts = {'validation': {'normal':20,'anomaly':20},
              'test': {'normal':30,'anomaly':30},
              'train': {'normal':200,'anomaly':40}}
    for label in ('normal','anomaly'):
        need = sum(x[label] for x in counts.values())
        if len(grouped[label]) < need:
            raise ValueError(f'Insuficientes {label}: se requieren {need}, disponibles {len(grouped[label])}.')
    output = []
    for label in ('normal','anomaly'):
        offset = 0
        for split in ('validation','test','train'):
            n = counts[split][label]
            for path in grouped[label][offset:offset+n]:
                output.append({'image':path.relative_to(ROOT).as_posix(), 'category':'pcb1', 'ground_truth':label, 'split':split})
            offset += n
    if len({x['image'] for x in output}) != len(output):
        raise AssertionError('Solapamiento entre conjuntos.')
    rng.shuffle(output)
    dest = ROOT / 'data' / 'pcb1_splits.csv'
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['image','category','ground_truth','split'])
        w.writeheader(); w.writerows(output)
    print(f'\nManifest: {dest}\n')
    for split in ('validation','test','train'):
        print(split, {label:sum(x['split']==split and x['ground_truth']==label for x in output) for label in ('normal','anomaly')})
    print('Exploration images excluded:',len(excluded))

if __name__ == '__main__': main()
