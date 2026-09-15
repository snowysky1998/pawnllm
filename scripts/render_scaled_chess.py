"""Render the completed scaled-model comparison and its validation learning curve."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--comparison', default='runs/chess-scaling/comparison-250k')
parser.add_argument('--training', default='runs/chess-human-250k/metrics.jsonl')
args = parser.parse_args()
root = Path(args.comparison)
report = json.loads((root / 'summary.json').read_text())
names = ['pawnllm_previous', 'pawnllm', 'maia3', 'stockfish', 'random']
labels = ['600-999', '1000-1399', '1400-1799', '1800-2199', '2200-2600', 'Overall']
fig, axes = plt.subplots(2, 1, figsize=(16, 10.8), dpi=160)
fig.patch.set_facecolor('#f6f8fb')
fig.text(.035, .955, 'Human-move prediction after scaling training', fontsize=24, weight='bold', color='#17243b')
fig.text(.035, .915, 'Each cell: N · geometric mean probability of the recorded move   |   Higher is better', fontsize=12, color='#47566c')
for ax, split, title in zip(axes, ['val', 'test'], ['July 2026 validation · checkpoint selection', 'August 2026 test · evaluated after selection']):
    ax.set_axis_off()
    ax.set_title(title, fontsize=15, loc='left', weight='bold', color='#17243b', pad=8)
    rows = [[label.replace('-', '–') if label != 'Overall' else 'All ratings'] +
            [f"{report['metrics'][split][name][label]['positions']:,} · {report['metrics'][split][name][label]['human_move_probability']:.2%}" for name in names]
            for label in labels]
    table = ax.table(cellText=rows, colLabels=['Player rating', 'Previous PawnLLM', 'PawnLLM · 250k games', 'Official Maia-3 23M', 'Stockfish + fitted\nprobabilities', 'Uniform random'],
                     cellLoc='center', colWidths=[.13, .17, .19, .18, .18, .15], bbox=[0, 0, 1, .97])
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#dbe2ec')
        cell.set_linewidth(.7)
        if i == 0:
            cell.set_facecolor('#17243b')
            cell.set_text_props(color='white', weight='bold', fontsize=10)
        else:
            cell.set_facecolor('#ffffff' if i % 2 else '#eef2f7')
            cell.set_text_props(color='#17243b', weight='bold' if i == 6 else 'normal')
            if j == 2:
                cell.set_facecolor('#dcefe6')
fig.text(.035, .075, 'New PawnLLM: fresh weights, pre-July training, FP32 policy head, native softmax. Other models retain their frozen previous settings.', fontsize=10.5, color='#47566c')
fig.text(.035, .047, 'One position per game · first 10 moves excluded · clocks ≥30s · equal Elo-band weight · baseline August scores were previously inspected', fontsize=10, color='#47566c')
fig.subplots_adjust(left=.035, right=.975, top=.86, bottom=.12, hspace=.24)
fig.savefig(root / 'comparison.png', facecolor=fig.get_facecolor())
fig.savefig(root / 'comparison.pdf', facecolor=fig.get_facecolor())
plt.close(fig)
events = [json.loads(line) for line in Path(args.training).read_text().splitlines()]
validation = [e for e in events if e['event'] == 'validation']
fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
ax.plot([e['step'] for e in validation], [100 * math.exp(-e['macro_policy_loss']) for e in validation], marker='o', label='New PawnLLM · July validation')
for name, label in [('pawnllm_previous', 'Previous PawnLLM'), ('stockfish', 'Stockfish + fitted probabilities'), ('maia3', 'Official Maia-3')]:
    ax.axhline(100 * report['metrics']['val'][name]['Overall']['human_move_probability'], linestyle='--', linewidth=1, label=label)
ax.set(xlabel='Optimizer steps', ylabel='Human-move probability (%)', title='250,000 training games · 1,950,536 positions')
ax.grid(alpha=.2)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(root / 'learning-curve.png')
plt.close(fig)
print(root / 'comparison.png')
