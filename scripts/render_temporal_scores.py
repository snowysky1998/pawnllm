"""Render fixed-model probabilities on July/August 2026 holdouts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('runs/temporal-scores')
r=json.loads((root/'summary.json').read_text())
names=['pawnllm','maia3','stockfish','random']
labels=['600-999','1000-1399','1400-1799','1800-2199','2200-2600','Overall']
fig,axes=plt.subplots(2,1,figsize=(14,10.8),dpi=160)
fig.patch.set_facecolor('#f6f8fb')
fig.text(.045,.958,'Human-move probability on the new holdouts',fontsize=24,weight='bold',color='#17243b')
fig.text(.045,.918,'Each cell: N · geometric mean probability of the recorded human move     |     Higher is better',fontsize=12,color='#47566c')
for ax,split,title in zip(axes,['val','test'],['Validation · July 2026 · 5,000 games','Test · August 2026 · 10,000 games']):
    ax.set_axis_off();ax.set_title(title,fontsize=15,loc='left',weight='bold',color='#17243b',pad=8)
    rows=[]
    for label in labels:
        cells=[label.replace('-','–') if label!='Overall' else 'All ratings']
        for name in names:
            s=r['metrics'][split][name][label]
            cells.append(f"{s['positions']:,} · {s['human_move_probability']:.2%}")
        rows.append(cells)
    t=ax.table(cellText=rows,colLabels=['Player rating','PawnLLM','Official Maia-3 23M','Stockfish + fitted\nprobabilities','Uniform random'],cellLoc='center',colWidths=[.15,.2,.24,.23,.18],bbox=[0,0,1,.97])
    t.auto_set_font_size(False);t.set_fontsize(11.5)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('#dbe2ec');cell.set_linewidth(.7)
        if i==0:cell.set_facecolor('#17243b');cell.set_text_props(color='white',weight='bold',fontsize=10.5)
        else:
            cell.set_facecolor('#ffffff' if i%2 else '#eef2f7');cell.set_text_props(color='#17243b')
            if i==6:cell.set_text_props(weight='bold')
    for i,label in enumerate(labels,1):
        best=max(range(4),key=lambda j:r['metrics'][split][names[j]][label]['human_move_probability'])+1
        t[i,best].set_facecolor('#dcefe6')
fig.text(.045,.078,'Weights and calibration stayed fixed. Games postdate the pinned Maia checkpoint’s May 17, 2026 publication.',fontsize=10.5,color='#47566c')
fig.text(.045,.049,'One position per game · first 10 moves excluded · clocks ≥30s · equal Elo-band weight · Stockfish 100k nodes / all legal moves',fontsize=10,color='#47566c')
fig.subplots_adjust(left=.045,right=.965,top=.864,bottom=.123,hspace=.24)
fig.savefig(root/'comparison.png',facecolor=fig.get_facecolor());fig.savefig(root/'comparison.pdf',facecolor=fig.get_facecolor())
plt.close(fig)
print(root/'comparison.png')
