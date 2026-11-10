"""Render metric-matched estimator outputs without recalibrating any method.

Figure contract: quantitative comparison grids. Fig. 5 displays cross-task
accuracy estimation and its availability limits. Fig. S14 separates ordinary
accuracy estimation from source-BA substitution evaluated against BA. Each point
is one model/cohort result, including five seeds for each of two backbones;
points are not independent patient replicates. No interval or inference is
invented. Missing estimates are counted, explained, and retained in source data.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ANALYSIS = Path(__file__).resolve().parents[1]
PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--input-dir', type=Path, default=ANALYSIS/'public_outputs/metric_matched')
PARSER.add_argument('--output-dir', type=Path, default=ANALYSIS/'results/revision_20261110/estimators')
PARSER.add_argument('--alignment-qa', action='store_true', help='Optional audit_panel_alignment module must already be on PYTHONPATH')
ARGS = PARSER.parse_args()
HERE = ARGS.output_dir.resolve()
DATA = ARGS.input_dir.resolve()
if HERE == DATA or DATA in HERE.parents:
    raise ValueError('Output must be outside the released input directory')
HERE.mkdir(parents=True, exist_ok=True)
if ARGS.alignment_qa:
    from audit_panel_alignment import require_matplotlib_panel_alignment


def alignment_check(fig, name):
    if ARGS.alignment_qa:
        return require_matplotlib_panel_alignment(fig, json_out=HERE/(name+'.alignment.json'),
            overlay_svg=HERE/(name+'.alignment.svg'), tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    report={'verdict':'NOT RUN','reason':'Optional alignment audit not requested; figure generation does not require external skill tools.'}
    (HERE/(name+'.alignment.json')).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report
METHODS = ['ALine-D', 'ALine-S', 'ATC-MC', 'DoC-Feat']
DISPLAY = {'ddr':'DDR', 'jsiec':'JSIEC', 'deepdrid':'DeepDRiD', 'aptos':'APTOS',
           'idrid':'IDRiD', 'messidor2':'Messidor-2', 'eyepacs':'EyePACS'}
COLORS = {'ddr':'#5B6B7A', 'jsiec':'#0072B2', 'deepdrid':'#D55E00',
          'aptos':'#009E73', 'idrid':'#CC79A7', 'messidor2':'#E69F00', 'eyepacs':'#56B4E9'}
MARKERS = {'dinov2_l':'o', 'retfound':'^'}
BACKBONES = {'dinov2_l':'DINOv2-L', 'retfound':'RETFound'}

plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                     'font.size':7, 'axes.labelsize':7, 'axes.titlesize':8,
                     'xtick.labelsize':6.5, 'ytick.labelsize':6.5,
                     'legend.fontsize':6.5, 'axes.linewidth':.6,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'xtick.major.width':.6, 'ytick.major.width':.6,
                     'xtick.major.size':2.5, 'ytick.major.size':2.5,
                     'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42,
                     'savefig.facecolor':'white'})


def read_rows(filename):
    with (DATA/filename).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def number(row, key):
    return float(row[key]) if row[key] else np.nan


def panel(ax, rows, protocol, truth_key, method, letter, limits, tick_values, small=False):
    x = np.array([number(r,truth_key) for r in rows])
    y = np.array([number(r,protocol+'_'+method) for r in rows])
    finite = np.isfinite(y)
    assert np.isfinite(x).all()
    # No filtering for convenience: only unidentifiable estimates lack a point.
    for cohort in dict.fromkeys(r['cohort'] for r in rows):
        for backbone, marker in MARKERS.items():
            mask = np.array([r['cohort']==cohort and r['backbone']==backbone for r in rows]) & finite
            ax.scatter(x[mask],y[mask],s=14 if small else 23,marker=marker,c=COLORS[cohort],
                       linewidths=.25,edgecolors='#FFFFFF',alpha=.9,zorder=3)
    diag_min,diag_max = max(limits[0][0],limits[1][0]), min(limits[0][1],limits[1][1],1.)
    ax.plot([diag_min,diag_max],[diag_min,diag_max],ls=(0,(3,3)),color='#777777',lw=.65,zorder=1)
    ax.set(xlim=limits[0],ylim=limits[1],xticks=tick_values,yticks=tick_values)
    ax.grid(False)
    ax.set_title(method,fontsize=7.5 if small else 8.5,pad=21)
    ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-15,22),textcoords='offset points',
                fontweight='bold',fontsize=8 if small else 9,ha='left',va='bottom')
    mae = float(np.mean(np.abs(y[finite]-x[finite])))
    ax.text(.5,1.025,f'{int(finite.sum())}/{len(rows)} available; MAE = {mae:.3f}',
            transform=ax.transAxes,fontsize=6.2 if small else 6.5,ha='center',va='bottom')
    assert (x[finite]>=limits[0][0]).all() and (x[finite]<=limits[0][1]).all()
    assert (y[finite]>=limits[1][0]).all() and (y[finite]<=limits[1][1]).all()
    return {'panel':letter,'method':method,'protocol':protocol,'target_metric':truth_key,
            'n_total':len(rows),'n_available':int(finite.sum()),'n_unavailable':int((~finite).sum()),
            'mae':mae,'raw_estimate_range':[float(np.min(y[finite])),float(np.max(y[finite]))],
            'unavailable_rows':[{'cohort':r['cohort'],'model':r['model']} for r,ok in zip(rows,finite) if not ok],
            'interval':'None: all model/cohort points are shown; no uncertainty interval or significance inference is claimed.',
            'x_limits':list(limits[0]),'y_limits':list(limits[1])}


def legend(fig, cohorts, y_cohort, y_backbone, columns):
    cohort_handles=[Line2D([],[],ls='',marker='o',markersize=4,markeredgewidth=0,color=COLORS[c],label=DISPLAY[c]) for c in cohorts]
    fig.legend(handles=cohort_handles,loc='upper center',bbox_to_anchor=(.53,y_cohort),ncol=columns,
               frameon=False,handletextpad=.4,columnspacing=1.15,labelspacing=.6)
    backbone_handles=[Line2D([],[],ls='',marker=MARKERS[b],markersize=4,markeredgewidth=0,color='#555555',label=BACKBONES[b]) for b in MARKERS]
    backbone_handles.append(Line2D([],[],ls=(0,(3,3)),color='#777777',lw=.65,label='Identity'))
    fig.legend(handles=backbone_handles,loc='upper center',bbox_to_anchor=(.53,y_backbone),ncol=3,
               frameon=False,handletextpad=.6,columnspacing=1.4)


def export(fig, name):
    fig.canvas.draw()
    alignment=alignment_check(fig,name)
    # Keep the declared physical page dimensions; no tight bounding-box rescale.
    fig.savefig(HERE/(name+'.pdf'))
    fig.savefig(HERE/(name+'.svg'))
    fig.savefig(HERE/(name+'_300dpi.png'),dpi=300)
    fig.savefig(HERE/(name+'_600dpi.png'),dpi=600)
    plt.close(fig)
    return alignment.get('verdict',alignment.get('status','see alignment JSON'))


def main():
    main_rows=read_rows('main_model_metrics.csv')
    same_rows=read_rows('same_task_model_metrics.csv')
    natural=[r for r in same_rows if r['setting']=='same_task_natural']
    assert len(main_rows)==70 and len(natural)==40
    summary=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    required={'setting','cohort','model','backbone','actual_accuracy','balanced_accuracy',
              'source_accuracy','source_balanced_accuracy','aline_pairs_retained','aline_design_rank','aline_d_status'}
    required.update(protocol+'_'+method for protocol in ['primary','source_ba'] for method in METHODS+['No-shift'])
    assert required <= set(main_rows[0]) and required <= set(same_rows[0]), 'Input schema changed'
    assert len(same_rows)==100
    for rows in [main_rows,same_rows]:
        assert len({(r['setting'],r['cohort'],r['model']) for r in rows})==len(rows)
    files=['same_task_model_metrics.csv','main_model_metrics.csv','summary.json','diagnostics.json','metadata.json','anchor_aggregate.json']
    input_hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in files}
    records=[]
    fig,axes=plt.subplots(2,2,figsize=(180/25.4,(180/1.095)/25.4))
    fig.subplots_adjust(left=.105,right=.98,bottom=.205,top=.85,wspace=.28,hspace=.62)
    fig.text(.53,.967,'Clinical anchor to public PDR targets',ha='center',fontsize=9,fontweight='bold')
    fig.text(.53,.941,'Ordinary accuracy; 10 models across 7 balanced target cohorts',ha='center',fontsize=7)
    for i,(ax,method) in enumerate(zip(axes.flat,METHODS)):
        rec=panel(ax,main_rows,'primary','actual_accuracy',method,'abcd'[i],((.45,1.05),(.25,1.18)),[.5,.75,1.])
        ax.set_yticks([.25,.5,.75,1.])
        np.testing.assert_allclose(rec['mae'],summary['main_balanced']['protocols']['primary'][method]['actual_accuracy']['mae'],atol=1e-12)
        ax.set_xlabel('Observed accuracy')
        ax.set_ylabel('Estimated accuracy')
        records.append({'figure':'Fig5',**rec})
    legend(fig,list(DISPLAY),.142,.083,4)
    fig.text(.53,.037,'ALine-D: all 10 Messidor-2 estimates unavailable (rank 9/10).',ha='center',fontsize=6.5)
    fig.text(.53,.018,'Raw DoC-Feat estimates above 1 are retained.',ha='center',fontsize=6.5)
    export(fig,'Fig5_metric_matched')

    fig,axes=plt.subplots(2,4,figsize=(180/25.4,120/25.4))
    fig.subplots_adjust(left=.09,right=.985,bottom=.20,top=.785,wspace=.33,hspace=1.02)
    fig.text(.53,.971,'Metric choice in the same-task control',ha='center',fontsize=9,fontweight='bold')
    fig.text(.53,.936,'40 model/cohort pairs at natural prevalence; DDR source anchor',ha='center',fontsize=7)
    fig.text(.53,.893,'Ordinary accuracy anchor and evaluation',ha='center',fontsize=7.5,fontweight='bold')
    fig.text(.53,.495,'Source-BA substitution and BA evaluation (adaptation)',ha='center',fontsize=7.5,fontweight='bold')
    for row,(protocol,truth) in enumerate([('primary','actual_accuracy'),('source_ba','balanced_accuracy')]):
        for col,method in enumerate(METHODS):
            ax=axes[row,col]
            rec=panel(ax,natural,protocol,truth,method,'abcdefgh'[row*4+col],((.45,1.04),(.45,1.04)),[.5,.75,1.],small=True)
            np.testing.assert_allclose(rec['mae'],summary['same_task_natural']['protocols'][protocol][method][truth]['mae'],atol=1e-12)
            ax.set_xlabel('Observed accuracy' if row==0 else 'Observed BA',fontsize=6.5)
            if col==0:
                ax.set_ylabel('Estimated value',fontsize=6.5)
            records.append({'figure':'FigS14',**rec})
    legend(fig,['aptos','eyepacs','messidor2','idrid'],.12,.075,4)
    fig.text(.53,.02,'BA = balanced accuracy. Bottom row is a source-metric adaptation, not a published BA estimator.',ha='center',fontsize=6)
    export(fig,'FigS14_metric_sensitivity')

    # Traceable plotting inputs, including NaNs and unplotted methods, copied as aggregate rows.
    for filename,rows in [('Fig5_source_data.csv',main_rows),('FigS14_source_data.csv',natural)]:
        with (HERE/filename).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    report={'backend':'Python/matplotlib', 'matplotlib':matplotlib.__version__,
            'physical_dimensions_mm':{'Fig5':[180,180/1.095],'FigS14':[180,120]},
            'minimum_declared_font_pt':6.0,'exports':['editable-vector PDF','editable SVG','300-dpi PNG','600-dpi PNG'],
            'archetype':'quantitative comparison grids',
            'figure_claims':{'Fig5':'Cross-task accuracy estimates and their availability differ from observed target accuracy; missing values and out-of-range raw estimates remain visible.',
                             'FigS14':'Source/evaluation metric choice changes the interpretation of the same-task natural-prevalence results.'},
            'data_policy':'All 70 main rows and 40 natural-prevalence rows retained in source data. No target-label tuning, confidence intervals, statistical tests or clipping. Missing ALine-D values explicitly counted.',
            'panels':records,'input_sha256':input_hashes,
            'rendered_visual_review':'Not performed by this script. Rendering and numeric assertions are not visual or PDF collision QA.'}
    (HERE/'figure_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Rendered Fig5 and FigS14 with all requested source rows; optional alignment QA:', ARGS.alignment_qa)


if __name__=='__main__':
    main()
