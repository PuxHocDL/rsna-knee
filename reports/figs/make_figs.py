# Figures for HPDA_solution_writeup.md. All numbers are copied from the write-up tables (no new measurements).
# usage: python make_figs.py   (writes fig*.png next to this file)
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
BLUE, ORANGE, RED, NEUTRAL = '#2a78d6', '#eb6834', '#e34948', '#f0efec'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                     'savefig.facecolor': SURFACE, 'axes.spines.top': False, 'axes.spines.right': False})


def finish(fig, name, title, subtitle):
    h = fig.get_figheight()   # place the header in inches so short and tall figures get the same spacing
    fig.subplots_adjust(top=1 - 1.05 / h)
    fig.text(0.02, 1 - 0.32 / h, title, ha='left', va='center', fontsize=14, fontweight='bold', color=INK)
    fig.text(0.02, 1 - 0.66 / h, subtitle, ha='left', va='center', fontsize=10.5, color=INK2)
    fig.savefig(OUT / name, dpi=200, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)


# ---------- 1. pipeline ----------
def box(ax, x, y, w, h, text, fc, ec, bold=False, size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, color=INK, fontweight='bold' if bold else 'normal',
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.3, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis('off')
PUB, OWN = '#eef4fc', '#fdeee6'
box(ax, 0.2, 2.6, 1.7, 1.0, 'MRI study\n(DICOM, all\nseries)', NEUTRAL, MUTED, size=10)
box(ax, 2.5, 4.55, 2.6, 0.9, 'Raptor CoAtNet-384\n(nartaa, OAI-trained)', PUB, BLUE)
box(ax, 2.5, 3.45, 2.6, 0.9, 'Raptor CoAtNet-224\n(nartaa)', PUB, BLUE)
box(ax, 2.5, 2.15, 2.6, 0.9, 'R7 ConvNeXt-S, 4 folds\n+ mirror TTA (heliosli)', PUB, BLUE)
box(ax, 2.5, 0.85, 2.6, 0.9, 's6fall (nartaa)', PUB, BLUE)
box(ax, 5.7, 3.55, 2.1, 1.75, 'Raptor slot\n0.65 · r384\n+ 0.35 · r224', '#ffffff', MUTED)
box(ax, 8.3, 2.2, 1.9, 2.0, 'Per-finding\nrank blend\nRaptor / R7\n0.5 / 0.5*', '#ffffff', MUTED)
box(ax, 8.3, 0.75, 1.9, 1.0, '× 2/3 + 1/3 ·\nrank(s6fall)', '#ffffff', MUTED)
box(ax, 10.75, 3.45, 2.05, 1.25, 'Synovitis only\nown B4 (EffV2-S)\nshare 25–50%', OWN, ORANGE)
box(ax, 10.75, 1.85, 2.05, 1.25, 'Fracture only\ndkim54 v11i\nshare 25%', PUB, BLUE)
box(ax, 10.75, 0.25, 2.05, 1.0, 'submission.csv\n12 columns', NEUTRAL, MUTED, bold=True)
for y in (5.0, 3.9, 2.6, 1.3):
    arrow(ax, 1.9, 3.1, 2.5, y)
arrow(ax, 5.1, 5.0, 5.7, 4.6); arrow(ax, 5.1, 3.9, 5.7, 4.2)
arrow(ax, 7.8, 4.4, 8.3, 3.7); arrow(ax, 5.1, 2.6, 8.3, 2.7)
arrow(ax, 9.25, 2.2, 9.25, 1.75); arrow(ax, 5.1, 1.3, 8.3, 1.25)
arrow(ax, 10.2, 1.25, 10.75, 3.7)
arrow(ax, 11.775, 1.85, 11.775, 1.25); arrow(ax, 11.775, 3.45, 11.775, 3.1)
ax.add_patch(FancyBboxPatch((0.25, 5.55), 0.32, 0.3, boxstyle='round,pad=0.01', fc=PUB, ec=BLUE, lw=1.4))
ax.text(0.68, 5.7, 'public weights', va='center', fontsize=10, color=INK2)
ax.add_patch(FancyBboxPatch((2.3, 5.55), 0.32, 0.3, boxstyle='round,pad=0.01', fc=OWN, ec=ORANGE, lw=1.4))
ax.text(2.73, 5.7, 'trained by us', va='center', fontsize=10, color=INK2)
ax.text(0.2, 0.25, '* heliosli per-finding Raptor weights on Lateral OA, PF OA, Synovitis, Baker\'s', fontsize=8.5, color=INK2)
finish(fig, 'fig1_pipeline.png', 'Final pipeline',
       'Every stage ranks each finding separately; optional stages fall back to the previous submission if the 9 h clock runs short')

# ---------- 2. leaderboard progression ----------
steps = ['10-10\nfirst\nblend', '10-11\npublic-reader\nblends', '10-13\n0.956 recipe\nreproduced', '10-14\n+ mirror TTA\n+ 224 ckpt',
         '10-15\n+ s6fall\n(G6R)', '10-17\nG6RM /\nG6RW', '10-18\n+ B4 as\nequal leg', '10-19\n+ B4 Syn 25%\n+ DK Frac 25%', '10-21\nB4 Syn 50%']
pub = [0.950, 0.954, 0.956, 0.956, 0.957, 0.958, 0.957, 0.958, 0.959]
prv = [0.933, 0.938, 0.940, 0.941, 0.942, 0.943, 0.943, 0.944, 0.944]
fig, ax = plt.subplots(figsize=(13, 5.6))
x = range(len(steps))
ax.plot(x, pub, color=BLUE, lw=2, marker='o', ms=8, mec=SURFACE, mew=2, label='Public LB', zorder=3)
ax.plot(x, prv, color=ORANGE, lw=2, marker='o', ms=8, mec=SURFACE, mew=2, label='Private LB', zorder=3)
ax.set_xticks(list(x)); ax.set_xticklabels(steps, fontsize=9.5)
ax.set_ylim(0.928, 0.963); ax.set_ylabel('Macro AUC (three decimals, as displayed)')
ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.text(8.15, pub[-1], 'Public 0.959', va='center', color=INK, fontsize=10.5)
ax.text(8.15, prv[-1], 'Private 0.944', va='center', color=INK, fontsize=10.5)
ax.text(0.1, prv[0] - 0.0025, '0.933', ha='center', color=INK2, fontsize=9.5)
ax.axvspan(5.5, 8.5, color=NEUTRAL, zorder=0)
ax.text(7.0, 0.9615, 'own components added', ha='center', color=INK2, fontsize=9.5)
ax.legend(loc='lower right', frameon=False)
ax.set_xlim(-0.4, 9.2)
finish(fig, 'fig2_progression.png', 'Leaderboard progression',
       'One change per step. Most of the private climb (0.933 → 0.943) came from assembling public readers; our own models added ~0.001')

# ---------- 3. labels ----------
rows = ['v3\n(fold 0)', 'v4h\n(fold 1)', 'v5h\n(mean of 4 single folds)', 'B4 = v5h folds 2,3,4,7\n(rank mean)']
gold = [0.9266, 0.9289, 0.9283, 0.9307]
ref = [0.9166, 0.9219, 0.9271, 0.9290]
fig, ax = plt.subplots(figsize=(11, 4.6))
y = list(range(len(rows)))[::-1]
for yi, g, r in zip(y, gold, ref):
    ax.plot([r, g], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle='round')
ax.scatter(gold, y, s=110, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3, label='gold-58 (radiologist reads)')
ax.scatter(ref, y, s=110, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3, label='ref392 (held out of training)')
for yi, r in zip(y, ref):
    ax.text(r - 0.0006, yi + 0.22, f'{r:.4f}', ha='center', fontsize=9, color=INK2)
for yi, g in zip(y, gold):
    ax.text(g + 0.0006, yi + 0.22, f'{g:.4f}', ha='center', fontsize=9, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(rows); ax.set_xlim(0.913, 0.934); ax.set_ylim(-0.6, 3.7)
ax.set_xlabel('Macro AUC'); ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(loc='lower left', frameon=False, fontsize=10)
finish(fig, 'fig3_labels.png', 'Model B by label version',
       'Mixing out-of-fold image predictions into report labels moved the held-out reference set most (+0.012 from v3 to B4)')

# ---------- 4. ablations ----------
items = [('Eval stride 1 instead of 3', 0.0007, 'ref392'), ('Label v10c, all 10 folds', 0.0008, 'ref-B'),
         ('Zoomed local views (fold 3)', -0.0029, 'ref392'), ('CoAtNet-2 trunk, best variant', -0.0060, 'ref392'),
         ('Robustness recipe R2 (EMA, BN recal, aug v2)', -0.0110, 'ref392'), ('Flip TTA on B4', -0.0140, 'ref392')]
fig, ax = plt.subplots(figsize=(11, 4.6))
yy = list(range(len(items)))[::-1]
for yi, (name, d, s) in zip(yy, items):
    ax.barh(yi, d, height=0.55, color=BLUE if d > 0 else RED, edgecolor=SURFACE, linewidth=2)
    ax.text(d + (0.0004 if d > 0 else -0.0004), yi, f'{d:+.4f}  ({s})', va='center', ha='left' if d > 0 else 'right',
            fontsize=9.5, color=INK2)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(yy); ax.set_yticklabels([n for n, _, _ in items])
ax.set_xlim(-0.0215, 0.006); ax.set_xlabel('Change in macro AUC vs the matching baseline (held-out sets only)')
ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
finish(fig, 'fig4_ablations.png', 'What moved our own model',
       'Only free inference settings and labels helped, and only by < 0.001; architecture and TTA changes all lost')

# ---------- 5. report ceiling ----------
srcs = [('Our hand re-read of the reports', 0.834), ('R6 out-of-fold (image model)', 0.924),
        ('lingyu07 v53 (report + image fusion)', 0.931), ('v5h (our training labels)', 0.935)]
fig, ax = plt.subplots(figsize=(11, 3.9))
yy = list(range(len(srcs)))[::-1]
ax.scatter([v for _, v in srcs], yy, s=130, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)   # dots: the axis does not start at 0
for yi, (_, v) in zip(yy, srcs):
    ax.text(v + 0.003, yi, f'{v:.3f}', va='center', fontsize=10, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([n for n, _ in srcs]); ax.set_xlim(0.80, 0.95); ax.set_ylim(-0.6, 3.6)
ax.set_xlabel('Macro AUC against expert reference labels (same 100 studies)')
ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
finish(fig, 'fig5_report_ceiling.png', 'Reports are not the target',
       'Reading reports more carefully scored far below labels that fuse in image-model evidence')
print('wrote', sorted(p.name for p in OUT.glob('fig*.png')))
