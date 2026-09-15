"""Render the saved human-prediction comparison (requires matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('runs/human-comparison')
r = json.loads((root / 'summary.json').read_text())
names = ['original_raw', 'refined_calibrated', 'stockfish_calibrated']
labels = ['600-999', '1000-1399', '1400-1799', '1800-2199', '2200-2600', 'Overall']
rows = []
for label in labels:
    cells = [label.replace('-', '–') if label != 'Overall' else 'All ratings\n(equal band weight)']
    for name in names:
        s = r['metrics'][name]['test'][label]
        cells.append(f"{s['positions']:,} · {s['macro_nll']:.3f}\nTop 3: {s['top3']:.2%}")
    rows.append(cells)
fig, ax = plt.subplots(figsize=(13.5, 8.3), dpi=160)
fig.patch.set_facecolor('#f6f8fb')
ax.set_axis_off()
fig.text(.055, .945, 'Predicting human chess moves', fontsize=23, weight='bold', color='#17243b')
fig.text(.055, .902, 'Test set · 3,840 positions / 480 games · five Lichess blitz rating bands', fontsize=12, color='#47566c')
fig.text(.055, .866, 'Each cell: N · human-move log loss ↓     |     Top-three agreement ↑', fontsize=12, color='#47566c')
t = ax.table(cellText=rows, colLabels=['Player rating', 'Original PawnLLM', 'Refined + calibrated', 'Stockfish + calibration'],
             cellLoc='center', colWidths=[.20, .265, .265, .27], bbox=[0, .20, 1, .75])
t.auto_set_font_size(False)
t.set_fontsize(12)
for (i, j), cell in t.get_celld().items():
    cell.set_edgecolor('#dbe2ec')
    cell.set_linewidth(.7)
    if i == 0:
        cell.set_facecolor('#17243b')
        cell.set_text_props(color='white', weight='bold', fontsize=11)
    else:
        cell.set_facecolor('#ffffff' if i % 2 else '#eef2f7')
        cell.set_text_props(color='#17243b')
        if i == 6:
            cell.set_text_props(weight='bold')
for i, label in enumerate(labels, 1):
    best = min(range(3), key=lambda j: r['metrics'][names[j]]['test'][label]['macro_nll']) + 1
    t[i, best].set_facecolor('#dcefe6')
fig.text(.055, .183, 'After the opening: Stockfish wins log loss (2.289 vs 2.320, equal band weight).', fontsize=12, weight='bold', color='#17243b')
fig.text(.055, .145, 'Stockfish 19: 100k nodes / position, every legal move; probabilities fitted on validation only.', fontsize=10, color='#47566c')
fig.text(.055, .117, 'The earlier 10k-node / top-3-only search was a different engine configuration.', fontsize=10, color='#47566c')
fig.text(.055, .089, 'Calibration: 4,016 validation positions. Training: 192,144 positions. Test set previously inspected.', fontsize=10, color='#47566c')
fig.text(.055, .061, 'Green marks lowest log loss. Overall top-3 is position-weighted. This is a fitted Stockfish baseline.', fontsize=10, color='#47566c')
fig.subplots_adjust(left=.055, right=.965, top=.85, bottom=.08)
fig.savefig(root / 'comparison.png', facecolor=fig.get_facecolor())
fig.savefig(root / 'comparison.pdf', facecolor=fig.get_facecolor())
plt.close(fig)
print(root / 'comparison.png')
