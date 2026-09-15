"""Render human-move probabilities including measured official Maia-3 results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('runs/human-comparison')
report = json.loads((root/'summary.json').read_text())
maia = json.loads(Path('runs/maia3-comparison/summary.json').read_text())
a = np.load(root/'scores.npz')
bands = ['600-999','1000-1399','1400-1799','1800-2199','2200-2600','Overall']
names = ['refined_calibrated','maia3_23m_official','stockfish_calibrated','uniform']
headers = ['Player rating','PawnLLM\nrefined + calibrated','Official Maia-3\n23M','Stockfish + fitted\nprobabilities','Uniform random']
metrics = {}
for split in ['val','test']:
    metrics[split] = {}
    rows = []
    for band in bands:
        mask = a[split].copy()
        if band != 'Overall':
            mask &= a['bands'] == bands.index(band)
        counts = np.isfinite(a['original'][mask]).sum(1)
        uniform_nll = np.mean([np.log(counts[a['bands'][mask] == b]).mean() for b in np.unique(a['bands'][mask])])
        metrics[split][band] = {}
        cells = [band.replace('-', '–') if band != 'Overall' else 'All ratings']
        for name in names:
            if name == 'uniform':
                nll, n = uniform_nll, int(mask.sum())
            else:
                data = maia if name == 'maia3_23m_official' else report
                stat = data['metrics'][name][split][band]
                nll, n = stat['macro_nll'], stat['positions']
            probability = float(np.exp(-nll))
            metrics[split][band][name] = dict(positions=n, human_move_probability=probability)
            cells.append(f'{n:,} · {probability:.2%}')
        rows.append(cells)
    fig, ax = plt.subplots(figsize=(14, 7.7), dpi=160)
    fig.patch.set_facecolor('#f6f8fb')
    ax.set_axis_off()
    fig.text(.04,.94,'How much probability did each model give the human’s move?',fontsize=19,weight='bold',color='#17243b')
    subtitle = 'Test: 3,840 positions from 480 games' if split == 'test' else 'Validation: 4,016 positions from 502 games (used to fit calibration)'
    fig.text(.04,.889,subtitle+' · Lichess blitz · higher is better',fontsize=13,color='#47566c')
    fig.text(.04,.848,'Each cell: N · human-move probability     |     Geometric mean, not move-matching accuracy',fontsize=12,color='#47566c')
    t = ax.table(cellText=rows,colLabels=headers,cellLoc='center',colWidths=[.16,.22,.20,.22,.20],bbox=[0,.24,1,.70])
    t.auto_set_font_size(False); t.set_fontsize(12)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('#dbe2ec');cell.set_linewidth(.7)
        if i == 0:
            cell.set_facecolor('#17243b');cell.set_text_props(color='white',weight='bold',fontsize=11)
        else:
            cell.set_facecolor('#ffffff' if i%2 else '#eef2f7');cell.set_text_props(color='#17243b')
            if i == 6: cell.set_text_props(weight='bold')
    for i,band in enumerate(bands,1):
        best=max(range(len(names)),key=lambda j:metrics[split][band][names[j]]['human_move_probability'])+1
        t[i,best].set_facecolor('#dcefe6')
    fig.text(.04,.195,'Maia-3 uses official released weights, actual player ratings, and eight-position history; no local fine-tuning.',fontsize=11,color='#17243b')
    fig.text(.04,.153,'Possible Maia training overlap: these games are held out from PawnLLM, not verified held out from Maia.',fontsize=11,color='#47566c')
    fig.text(.04,.114,'All-ratings score gives each Elo band equal weight. Green marks the highest probability in each row.',fontsize=10.5,color='#47566c')
    fig.text(.04,.075,'PawnLLM / Stockfish calibration fitted on validation only. Official Maia probabilities are shown without added calibration.',fontsize=10.5,color='#47566c')
    fig.subplots_adjust(left=.04,right=.97,top=.82,bottom=.06)
    suffix = '' if split == 'test' else '-validation'
    fig.savefig(root/f'human-move-probability{suffix}.png',facecolor=fig.get_facecolor())
    fig.savefig(root/f'human-move-probability{suffix}.pdf',facecolor=fig.get_facecolor())
    plt.close(fig)
(root/'human-move-probability.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(root/'human-move-probability.png')
