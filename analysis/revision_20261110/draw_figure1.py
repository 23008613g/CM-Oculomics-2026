"""Render the study-design flowchart from reported aggregate study facts.

All symbols are geometric schematic elements, not clinical images. No data
sampling, model inference or treatment decision is represented by the artwork.
The public DR-grading proxy is kept distinct from the clinical outcome.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle

WIDTH_MM = 183
HEIGHT_MM = 126
INK = "#263238"
MUTED = "#596772"
LINE = "#BAC6CC"
BLUE = "#385F7C"
TEAL = "#367C7C"
WARM = "#A26449"
PALE_BLUE = "#F0F5F8"
PALE_TEAL = "#EDF5F4"
PALE_WARM = "#FAF1EB"


def text(ax, x, y, label, size=7, color=INK, weight="normal", ha="left", va="top"):
    return ax.text(x, y, label, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, linespacing=1.3)


def box(ax, x, y, w, h, fill="white", edge=LINE, radius=1.2, lw=.55):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={radius}",
                 facecolor=fill, edgecolor=edge, linewidth=lw))


def arrow(ax, start, end, color=MUTED, lw=.65):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                 mutation_scale=6, linewidth=lw, color=color,
                 shrinkA=.7, shrinkB=.7))


def line(ax, points, color=LINE, lw=.6):
    ax.plot(*zip(*points), color=color, linewidth=lw, solid_capstyle="round")


def camera(ax, x, y, color=TEAL):
    box(ax, x, y, 8.5, 5.8, edge=color, radius=.6, lw=.7)
    ax.add_patch(Circle((x+4.5, y+2.9), 1.8, fill=False, edgecolor=color, linewidth=.7))
    line(ax, [(x+1.3, y+5.8), (x+2.1, y+7), (x+5.2, y+7), (x+6, y+5.8)], color, .7)


def patches(ax, x, y, color):
    for row in range(3):
        for col in range(3):
            ax.add_patch(Rectangle((x+col*1.65, y+row*1.65), 1.2, 1.2,
                         linewidth=.45, edgecolor=color,
                         facecolor=color if row == col else "white"))


def panel(fig, letter, title, rect):
    x, y, w, h = rect
    ax = fig.add_axes([x/WIDTH_MM, y/HEIGHT_MM, w/WIDTH_MM, h/HEIGHT_MM], label=letter)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    text(ax, 0, h, letter, size=9, weight="bold")
    text(ax, 5.3, h-.1, title, size=8, weight="bold")
    return ax


def build_figure1(output_dir, alignment_qa=False, stem="Figure1_study_design_corrected"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "savefig.facecolor": "white", "axes.linewidth": .6})
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), facecolor="white")
    a = panel(fig, "a", "Clinical cohort and treatment-response endpoint", (5, 92, 173, 29))
    b = panel(fig, "b", "Image-only modelling and internal validation", (5, 42, 93, 44))
    c = panel(fig, "c", "Independent external evaluation", (105, 42, 73, 44))
    d = panel(fig, "d", "Transfer assessment and local validation", (5, 5, 173, 30))

    # a: chronology only. Pretreatment photographs, not post-treatment images,
    # are the prediction inputs; the model does not determine treatment.
    box(a, 0, 1.5, 35, 20.5, fill=PALE_BLUE, edge="#D7E2E8")
    text(a, 3, 19.8, "Clinical cohort", 7, BLUE, "bold")
    text(a, 3, 15.5, "489 patients", 8.8, INK, "bold")
    text(a, 3, 11, "988 photographs", 7.3)
    text(a, 3, 5.7, "2021–2024 · 3 camera setups", 6.2, MUTED)
    arrow(a, (36, 11.5), (41, 11.5))
    camera(a, 43, 12.3)
    text(a, 54, 20, "Pretreatment", 7.2, TEAL, "bold")
    text(a, 54, 16, "photograph", 6.8)
    text(a, 43, 9.2, "Baseline PDR", 7, INK, "bold")
    text(a, 43, 5.3, "NVE/NVD present", 6.5, MUTED)
    arrow(a, (75, 11.5), (80, 11.5))
    text(a, 102.5, 21, "Six monthly ranibizumab", 7, INK, "bold", ha="center")
    text(a, 102.5, 17, "injections", 6.8, ha="center")
    line(a, [(84, 10.5), (121, 10.5)], TEAL, .8)
    for i in range(6):
        x = 84+i*7.4
        a.add_patch(Circle((x, 10.5), 1, facecolor="white", edgecolor=TEAL, linewidth=.75))
        text(a, x, 7.6, str(i+1), 6.2, MUTED, ha="center")
    line(a, [(123, 10.5), (127.5, 10.5), (127.5, 17.5)], MUTED, .65)
    line(a, [(127.5, 10.5), (127.5, 6)], MUTED, .65)
    arrow(a, (127.5, 17.5), (132, 17.5))
    arrow(a, (127.5, 6), (132, 6))
    box(a, 133, 13, 40, 10, fill=PALE_WARM, edge="#DFC6B8")
    text(a, 153, 20.7, "Persistent/recurrent\nneovascularization", 6.7, WARM, "bold", ha="center")
    box(a, 133, 1.5, 40, 9, fill=PALE_TEAL, edge="#BFD7D3")
    text(a, 153, 7.9, "Documented regression", 6.6, TEAL, "bold", ha="center")

    # b: the linked subset, not the full photograph cohort, is partitioned.
    patches(b, 0, 31.5, BLUE)
    text(b, 7.4, 36.5, "DINOv2", 8.2, BLUE, "bold")
    text(b, 7.4, 32, "ViT-L/14 · primary", 6.5, MUTED)
    patches(b, 49, 31.5, TEAL)
    text(b, 56.4, 36.5, "RETFound", 8.2, TEAL, "bold")
    text(b, 56.4, 32, "ViT-L/16 · comparator", 6.5, MUTED)
    text(b, 46.5, 25.9, "Linked subset · 465 patients / 945 images", 7, INK, "bold", ha="center")
    line(b, [(46.5, 21.6), (46.5, 20), (21.5, 20), (21.5, 18.4)], MUTED)
    line(b, [(46.5, 20), (71.5, 20), (71.5, 18.4)], MUTED)
    arrow(b, (21.5, 18.9), (21.5, 17))
    arrow(b, (71.5, 18.9), (71.5, 17))
    for x, title, people, images, positives in [
        (0, "Five-fold cross-validation", 396, 805, 71),
        (50, "Held-out test", 69, 140, 11),
    ]:
        box(b, x, 2.6, 43, 14.5, fill=PALE_BLUE, edge="#D7E2E8")
        text(b, x+21.5, 14.9, title, 7, BLUE, "bold", ha="center")
        text(b, x+21.5, 10.8, f"{people} patients · {images} images", 6.7, ha="center")
        text(b, x+21.5, 6.7, f"{positives} outcome-positive images", 6.3, MUTED, ha="center")
    text(b, 46.5, .5, "Patient-disjoint partition", 6.3, MUTED, ha="center")

    # c: clinically trained models are frozen. The incoming public cohorts
    # have an independent grading task, without anti-VEGF response labels.
    text(c, 0, 36.5, "Frozen clinical models", 8.2, BLUE, "bold")
    text(c, 0, 32, "2 backbones × 5 training seeds", 6.8)
    arrow(c, (9, 29), (9, 26.5))
    text(c, 13, 29.8, "No further fine-tuning", 6.5, MUTED)
    box(c, 0, 2.6, 73, 23.5, fill=PALE_TEAL, edge="#BFD7D3")
    text(c, 36.5, 23.4, "7 cohorts · 3,016 images", 8.3, TEAL, "bold", ha="center")
    text(c, 36.5, 18.5, "Class-balanced PDR vs NPDR proxy", 6.8, ha="center")
    text(c, 36.5, 14.1, "DDR · JSIEC · DeepDRiD · APTOS", 6.6, ha="center")
    text(c, 36.5, 10, "IDRiD · Messidor-2 · EyePACS", 6.6, ha="center")
    text(c, 36.5, 5.9, "No treatment-outcome labels", 6.3, MUTED, ha="center")
    fig.add_artist(FancyArrowPatch((99/WIDTH_MM, 76/HEIGHT_MM), (103.5/WIDTH_MM, 76/HEIGHT_MM),
                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6,
                   color=MUTED, linewidth=.65, shrinkA=.5, shrinkB=.5))

    # d: parallel assessment components, not a serial estimator-training flow.
    line(d, [(54, 2), (54, 22)], "#DCE3E6", .55)
    line(d, [(115, 2), (115, 22)], "#DCE3E6", .55)
    text(d, 0, 21.9, "Label-free shift signals", 7.7, BLUE, "bold")
    text(d, 0, 16.3, "Five measures of shift,\nagreement and confidence", 7)
    text(d, 0, 5.8, "Association with external AUC", 6.6, MUTED)
    text(d, 60, 21.9, "Performance estimation", 7.7, BLUE, "bold")
    text(d, 60, 16.3, "ALine-D · ALine-S\nATC-MC · DoC-Feat", 7)
    text(d, 60, 5.8, "Clinical held-out anchor + separate\npublic same-task control", 6.3, MUTED)
    text(d, 121, 21.9, "Local validation requirements", 7.7, BLUE, "bold")
    text(d, 121, 16.3, "Resampling of public predictions", 6.7)
    text(d, 121, 11.3, "Balanced accuracy and AUC", 6.7)
    text(d, 121, 5.8, "Labels needed for specified precision", 6.3, MUTED)

    fig.canvas.draw()
    if alignment_qa:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=output_dir/f"{stem}.alignment.json",
            overlay_svg=output_dir/f"{stem}.alignment.svg", tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
            row_groups=[["b", "c"]], column_groups=[["a", "d"]])
    else:
        (output_dir/f"{stem}.alignment.json").write_text(json.dumps({
            "verdict": "NOT RUN", "reason": "Optional external alignment auditor was not requested."}, indent=2)+"\n", encoding="utf-8")
    fig.savefig(output_dir/f"{stem}.pdf")
    fig.savefig(output_dir/f"{stem}.svg")
    fig.savefig(output_dir/f"{stem}.png", dpi=600)
    fig.savefig(output_dir/f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(output_dir/f"{stem}.preview.png", dpi=300)
    plt.close(fig)
    return output_dir/f"{stem}.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1]/"results/revision_20261110/schematics")
    parser.add_argument("--alignment-qa", action="store_true", help="Requires the external audit_panel_alignment module on PYTHONPATH")
    args = parser.parse_args()
    print(build_figure1(args.output_dir, args.alignment_qa))
