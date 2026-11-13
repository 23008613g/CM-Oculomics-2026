"""Scientific schematics from verified study facts and the local app code.

Current study schematics; not a screenshot or new model inference.
No manuscript or clinical-image files are inputs. Fixed aggregate facts and
recorded example scores are rendered without new inference.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ANALYSIS=Path(__file__).resolve().parents[1]
ROOT=ANALYSIS.parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ANALYSIS/'results/revision_20261110/schematics')
parser.add_argument('--alignment-qa',action='store_true',help='Also run the bundled geometric alignment audit on Figures S1 and S7')
args=parser.parse_args()
BASE=args.output_dir.resolve()
BASE.mkdir(parents=True,exist_ok=True)
if args.alignment_qa:
    from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})
COLORS = {
    'blue': ('#EDF4FA', '#32678C'), 'orange': ('#FCF0E7', '#A45123'),
    'green': ('#EDF7F1', '#287254'), 'purple': ('#F3EFF8', '#745493'),
    'gold': ('#FCF7E9', '#866B28'), 'grey': ('#F3F5F6', '#52616C'),
}

def setup(ax, letter, title, size=7.5):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(0, 1, letter, fontsize=8.5, fontweight='bold', va='top')
    ax.text(.065, 1, title, fontsize=size, fontweight='bold', va='top', color='#263746')

def card(ax, x, y, w, h, title, body, color='blue', title_size=7, body_size=6.4,
         center=False, title_y=.09, body_y=.25):
    fill, edge = COLORS[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.012',
                               facecolor=fill, edgecolor=edge, linewidth=.65))
    tx = x + w / 2 if center else x + .045 * w
    align = 'center' if center else 'left'
    ax.text(tx, y + h * (1-title_y), title, ha=align, va='top', fontsize=title_size,
            fontweight='bold', color=edge, linespacing=1.15)
    if body:
        ax.text(tx, y + h * (1-body_y), body, ha=align, va='top', fontsize=body_size,
                color='#263238', linespacing=1.25)

def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=7,
                               color='#657783', linewidth=.7, shrinkA=1.5, shrinkB=1.5))

def figure_arrow(fig, a, b):
    fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle='-|>',
                                  mutation_scale=8, color='#657783', linewidth=.75,
                                  shrinkA=2, shrinkB=2))

def export(fig, name):
    fig.canvas.draw()
    if args.alignment_qa:
        require_matplotlib_panel_alignment(fig,
            json_out=str(BASE / f'{name}.alignment.json'),
            overlay_svg=str(BASE / f'{name}.alignment.svg'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            require_panel_labels=True, strict=True)
    else:
        (BASE / f'{name}.alignment.json').write_text(json.dumps({
            'verdict':'NOT RUN',
            'reason':'Optional alignment audit not requested; external skill tools are not rendering dependencies.'
        },indent=2)+'\n',encoding='utf-8')
    fig.savefig(BASE / f'{name}.pdf')
    fig.savefig(BASE / f'{name}.svg')
    fig.savefig(BASE / f'{name}.png', dpi=600)
    print(name, [round(float(v)*25.4, 3) for v in fig.get_size_inches()])
    plt.close(fig)


# Figure 1 uses the current 183 x 126 mm study-design flowchart.
# Isolate its style settings so the established S1/S7 exports remain unchanged.
from draw_figure1 import build_figure1
with plt.rc_context():
    build_figure1(BASE, BASE / 'qa', stem='Figure1_study_design_corrected')


# Figure S1: explicitly prospective hypothesis; no management recommendation.
fig = plt.figure(figsize=(166/25.4, 105/25.4))
gs = fig.add_gridspec(3, 2, height_ratios=[.75, 1.15, 1.15],
                      left=.035, right=.98, top=.92, bottom=.035,
                      hspace=.25, wspace=.13)
fig.suptitle('Proposed imaging and constitutional framework', y=.985,
             fontsize=9.5, fontweight='bold')
a = fig.add_subplot(gs[0, :])
b, c = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
d = fig.add_subplot(gs[2, :])
setup(a, 'a', 'Research inputs', size=8)
card(a, 0, 0, 1, .73, 'Pretreatment fundus photograph + clinical and constitutional data',
     'NVE/NVD already present before the first ranibizumab injection',
     title_size=7.5, body_size=7.2, body_y=.55, title_y=.15, center=True)
setup(b, 'b', 'Imaging axis', size=8)
card(b, 0, 0, 1, .79, 'DINOv2 research score',
     'Post-treatment persistence/recurrence\nof neovascularization after six injections\n'
     'Score is not a calibrated probability', color='orange',
     title_size=7.5, body_size=7.2, body_y=.35)
setup(c, 'c', 'Constitutional axis', size=8)
card(c, 0, 0, 1, .79, 'Syndrome characterization',
     'Clinical and constitutional variables\nSupplementary Note S1', color='green',
     title_size=7.5, body_size=7.2, body_y=.35)
setup(d, 'd', 'Proposed combination and prospective evaluation', size=8)
card(d, 0, 0, 1, .79, 'Combined profile: imaging score + syndrome characterization',
     'Incremental prediction requires prospective evaluation.\n'
     'In this cohort, syndrome did not improve prediction; persistence/recurrence rates\n'
     'did not differ by syndrome. No management recommendation is made.',
     color='purple', title_size=7.5, body_size=7.2, body_y=.32, center=True)
for target in (b, c):
    p, q = a.get_position(), target.get_position()
    figure_arrow(fig, ((q.x0+q.x1)/2, p.y0), ((q.x0+q.x1)/2, q.y1+.006))
for source in (b, c):
    p, q = source.get_position(), d.get_position()
    figure_arrow(fig, ((p.x0+p.x1)/2, p.y0), ((p.x0+p.x1)/2, q.y1+.006))
export(fig, 'FigureS1_framework_corrected')


# Figure S7: documentation schematic, not a replacement UI screenshot.
fig = plt.figure(figsize=(166/25.4, 112/25.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], left=.035, right=.98,
                      top=.885, bottom=.16, hspace=.18, wspace=.13)
fig.suptitle('CM-Oculomics: implementation schematic', y=.985,
             fontsize=9.5, fontweight='bold')
fig.text(.5, .94, 'Local code pathway and previously recorded score examples',
         ha='center', va='top', fontsize=7.5, color='#52616C')
a = fig.add_subplot(gs[0, :])
b, c = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
setup(a, 'a', 'Local application pipeline', size=8)
nodes = [
    ('Input', 'Color fundus\nphotograph\nRGB image', 'blue'),
    ('Preprocessing', 'Resize to 224 × 224\nImageNet\nnormalization', 'blue'),
    ('DINOv2 ViT-L/14', 'Fine-tuned checkpoint\nTwo-class softmax\nClass 1 score', 'orange'),
    ('Model outputs', 'Research score\nPredicted-class\nGrad-CAM', 'green'),
]
node_w, gap = .22, .04
for i, (title, body, color) in enumerate(nodes):
    x = i * (node_w + gap)
    card(a, x, .29, node_w, .51, title, body, color,
         title_size=7, body_size=6.9, body_y=.34, center=True)
    if i < 3:
        arrow(a, (x+node_w, .54), (x+node_w+gap, .54))
a.text(.5, .17, 'Parallel image analysis: vascular density, skeleton length and fractal dimension',
       ha='center', va='center', fontsize=7, color='#263238')
a.text(.5, .04, 'Fine-tuned weights required; configurable score threshold (default 0.5)',
       ha='center', va='center', fontsize=7, color='#52616C')
for ax, letter, label, score, higher in [(b, 'b', 'Recorded example A', .397, False),
                                       (c, 'c', 'Recorded example B', .971, True)]:
    setup(ax, letter, label, size=8)
    color = 'orange' if higher else 'green'
    fill, edge = COLORS[color]
    ax.add_patch(FancyBboxPatch((0, 0), 1, .81, boxstyle='round,pad=0,rounding_size=.012',
                               facecolor=fill, edgecolor=edge, linewidth=.65))
    ax.text(.5, .70, f'{score:.3f}', fontsize=21, fontweight='bold', ha='center', va='top', color=edge)
    ax.text(.5, .38, 'Higher score — above threshold' if higher else 'Lower score — below threshold',
            ha='center', va='center', fontsize=7.5, color=edge)
    ax.text(.5, .22, 'Model class at the default threshold:', ha='center', va='center', fontsize=7)
    ax.text(.5, .08, 'NV persistence/recurrence' if higher else 'Documented regression',
            ha='center', va='center', fontsize=7.5, fontweight='bold', color=edge)
fig.text(.5, .105, 'Previously recorded scores; no new inference was performed for this schematic.',
         ha='center', fontsize=7, color='#52616C')
fig.text(.5, .068, 'Development endpoint: NV persistence/recurrence versus regression after six ranibizumab injections.',
         ha='center', fontsize=7, color='#52616C')
fig.text(.5, .032, 'NVE/NVD was already present in pretreatment photographs. Research score, not a calibrated probability.',
         ha='center', fontsize=7, color='#52616C')
export(fig, 'FigureS7_implementation_schematic')

provenance = {
    'clinical_endpoint': 'Study endpoint: NVE/NVD already present at first-injection baseline and in pretreatment model inputs; outcome after six monthly ranibizumab injections is persistence/recurrence versus documented regression.',
    'counts': 'Study aggregate counts embedded in the schematic; no patient-level files read.',
    'recorded_score_examples': [0.397, 0.971],
    'score_source': 'Existing Figure S7; no new model inference.',
    'threshold': 0.5,
    'threshold_source': 'Local app.py THRESHOLD default; configurable.',
    'source_hashes': {},
}
for relative in ['analysis/revision_20261110/draw_figure1.py',
                 'analysis/revision_20261110/draw_clinical_schematics.py',
                 'analysis/revision_20261110/audit_panel_alignment.py',
                 'app.py', 'src/dataset.py', 'src/biomarkers.py']:
    p = ROOT / relative
    provenance['source_hashes'][relative] = hashlib.sha256(p.read_bytes()).hexdigest()
(BASE / 'schematic_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
