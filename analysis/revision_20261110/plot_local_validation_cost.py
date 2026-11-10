"""Redraw Figure 6/S6 from the fully recomputed, unmodified curve cache.

Only the 0.10 and 0.05 error tolerances are shown. No estimator-performance
benchmark is plotted. The cache preserves every original simulated curve value.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ANALYSIS=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-dir',type=Path,default=ANALYSIS/'public_outputs/local_validation_revision')
parser.add_argument('--output-dir',type=Path,default=ANALYSIS/'results/revision_20261110/local_validation_figures')
parser.add_argument('--alignment-qa',action='store_true',help='Optional audit_panel_alignment module must already be on PYTHONPATH')
args=parser.parse_args()
DATA=args.input_dir.resolve(); BASE=args.output_dir.resolve()
if BASE==DATA or DATA in BASE.parents:
    raise ValueError('Output must be outside the released curve directory')
BASE.mkdir(parents=True,exist_ok=True)
if args.alignment_qa:
    from audit_panel_alignment import require_matplotlib_panel_alignment

validation = json.loads((DATA / 'cache_metadata.json').read_text(encoding='utf-8'))
assert validation['aggregate_exact_match']
GRID = np.asarray(validation['grid'])
assert np.all(GRID > 0) and np.all(np.diff(GRID) > 0)
PREV = [0.5, 0.2, 0.1, 0.088]
COLS = {0.5: '#999999', 0.2: '#56B4E9', 0.1: '#0072B2', 0.088: '#D55E00'}
CACHE = np.load(DATA / 'local_validation_curves.npz', allow_pickle=False)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})


def panel(ax, metric, backbone, xmode, title, letter):
    for prevalence in PREV:
        key = f'{metric}|{backbone}|{prevalence}'
        matrix = CACHE[key].copy()
        assert matrix.shape == (7, len(GRID))
        # Same display rule as the supplied script: an absent class can make
        # the error quantile nonfinite. Cache retains all original values.
        matrix[~np.isfinite(matrix)] = np.nan
        x = GRID * (prevalence if xmode == 'positive' else 1)
        assert np.all(x > 0)
        line, = ax.plot(x, np.nanmedian(matrix, axis=0), color=COLS[prevalence],
                        linewidth=1.2, label=f'{100 * prevalence:g}%')
        band = ax.fill_between(x, np.nanpercentile(matrix, 25, axis=0),
                               np.nanpercentile(matrix, 75, axis=0),
                               color=COLS[prevalence], alpha=0.15, linewidth=0)
        line._explicit_axes_clip = band._explicit_axes_clip = True
    ax.axhline(0.10, color='#777777', linewidth=0.7, linestyle='--')
    ax.axhline(0.05, color='#777777', linewidth=0.7, linestyle=':')
    ax.set_xscale('log')
    ax.set_ylim(0, 0.35)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel('Expected PDR-positive images' if xmode == 'positive' else 'Labeled images')
    ax.set_title(title, pad=5)
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-9, 5),
                textcoords='offset points', ha='left', va='baseline',
                fontsize=9, fontweight='bold', annotation_clip=False)


def explicitly_clip_data_paths(fig):
    """Apply the existing axes clip to exported geometry before PDF inspection.

    PDF drawing inventories can include invisible, clip-masked curve portions.
    Clip in display coordinates so log-x segment intersections are unchanged.
    This affects only geometry outside the existing axes range, never cached data.
    """
    for ax in fig.axes:
        box = ax.bbox
        inverse = ax.transData.inverted()
        for line in ax.lines:
            if not getattr(line, '_explicit_axes_clip', False):
                continue
            vertices = ax.transData.transform(line.get_xydata())
            kept = []
            for start, end in zip(vertices[:-1], vertices[1:]):
                if not np.all(np.isfinite([start, end])):
                    continue
                delta = end - start
                lo, hi = 0., 1.
                for p, q in zip((-delta[0], delta[0], -delta[1], delta[1]),
                                (start[0] - box.x0, box.x1 - start[0],
                                 start[1] - box.y0, box.y1 - start[1])):
                    if p == 0:
                        if q < 0:
                            lo, hi = 1., 0.
                            break
                    elif p < 0:
                        lo = max(lo, q / p)
                    else:
                        hi = min(hi, q / p)
                if lo <= hi:
                    a, b = start + lo * delta, start + hi * delta
                    if not kept or not np.allclose(kept[-1], a, rtol=0, atol=1e-9):
                        if kept:
                            kept.append(np.array([np.nan, np.nan]))
                        kept.append(a)
                    kept.append(b)
            values = inverse.transform(np.asarray(kept))
            line.set_data(values[:, 0], values[:, 1])
        for band in ax.collections:
            if not getattr(band, '_explicit_axes_clip', False):
                continue
            polygons = []
            for path in band.get_paths():
                clipped = path.transformed(ax.transData).clip_to_bbox(box)
                polygons.extend(inverse.transform(p) for p in clipped.to_polygons())
            band.set_verts(polygons)


def export(fig, name):
    fig.canvas.draw()
    # Preserve an exact raster comparison for the explicit vector-geometry clip.
    before_clip = np.asarray(fig.canvas.buffer_rgba()).copy()
    explicitly_clip_data_paths(fig)
    fig.canvas.draw()
    after_clip = np.asarray(fig.canvas.buffer_rgba())
    difference = np.abs(before_clip.astype(int) - after_clip.astype(int))
    changed_fraction = float(np.any(difference > 0, axis=2).mean())
    # Antialiasing at the axes boundary can differ by a subpixel.
    assert changed_fraction < 0.005, changed_fraction
    (BASE / f'{name}.clip-validation.json').write_text(json.dumps({
        'display_coordinate_clipping': True,
        'raster_changed_pixel_fraction': changed_fraction,
        'mean_absolute_channel_difference': float(difference.mean()),
        'cache_modified': False,
    }, indent=2))
    if args.alignment_qa:
        require_matplotlib_panel_alignment(
            fig, json_out=str(BASE / f'{name}.alignment.json'),
            overlay_svg=str(BASE / f'{name}.alignment.svg'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            require_panel_labels=True, strict=True)
    else:
        (BASE / f'{name}.alignment.json').write_text(json.dumps({
            'verdict':'NOT RUN',
            'reason':'Optional alignment audit not requested; external skill tools are not rendering dependencies.'
        }, indent=2)+'\n',encoding='utf-8')
    # Do not crop: the explicit 180-mm physical dimensions are authoritative.
    fig.savefig(BASE / f'{name}.pdf')
    fig.savefig(BASE / f'{name}.svg')
    fig.savefig(BASE / f'{name}.png', dpi=600)
    print(name, 'mm:', [round(x * 25.4, 3) for x in fig.get_size_inches()])
    plt.close(fig)


width = 180 / 25.4
fig, axs = plt.subplots(1, 2, figsize=(width, width / 2.6), sharey=True)
fig.subplots_adjust(left=0.10, right=0.985, bottom=0.19, top=0.76, wspace=0.20)
panel(axs[0], 'ba', 'DINOv2', 'images', 'By labeled images', 'a')
panel(axs[1], 'ba', 'DINOv2', 'positive', 'By expected PDR-positive images', 'b')
for ax in axs:
    # 9-pt parents give 6.3-pt log superscripts: 5.33 pt at Word's 152.4-mm width.
    ax.tick_params(axis='x', labelsize=9)
axs[0].set_ylabel('95th-percentile error\nof balanced accuracy')
fig.suptitle('DINOv2 ensemble: median and interquartile range across seven cohorts',
             fontsize=8, y=0.985)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.55, 0.91),
           ncol=4, handlelength=2.0, columnspacing=1.4, borderaxespad=0)
export(fig, 'Figure6_local_validation_cost')

fig, axs = plt.subplots(1, 3, figsize=(width, width / 3.5), sharey=True)
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.25, top=0.76, wspace=0.21)
panel(axs[0], 'ba', 'RETFound', 'images', 'RETFound, balanced accuracy', 'a')
panel(axs[1], 'auc', 'DINOv2', 'images', 'DINOv2, AUC', 'b')
panel(axs[2], 'auc', 'RETFound', 'images', 'RETFound, AUC', 'c')
axs[0].set_ylabel('95th-percentile error')
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.54, 0.985),
           ncol=4, handlelength=2.0, columnspacing=1.4, borderaxespad=0)
export(fig, 'FigureS6_local_validation_cost')
