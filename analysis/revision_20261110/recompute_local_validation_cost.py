"""Recompute public-cohort sampling requirements and complete error curves.

This is a sampling simulation from public PDR/NPDR prediction distributions,
not a validation of clinical treatment-outcome sample requirements. Images are
sampled with replacement and positive counts are binomial. Draws lacking a
class count as failures. No estimator-MAE benchmark is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

ANALYSIS = Path(__file__).resolve().parents[1]
COHORTS = ['ddr','jsiec','deepdrid','aptos','idrid','messidor2','eyepacs']
BACKBONES = {'dinov2_l':'DINOv2','retfound':'RETFound'}
PREVALENCES = [0.5,0.2,0.1,0.088]
GRID = np.unique(np.round(np.logspace(np.log10(20),np.log10(4000),40)).astype(int))
DRAWS_BA, DRAWS_AUC = 4000, 600


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ANALYSIS/'public_outputs/external_predictions')
    parser.add_argument('--reference',type=Path,default=ANALYSIS/'public_outputs/local_validation_revision/local_validation_cost.json')
    parser.add_argument('--output-dir',type=Path,default=ANALYSIS/'results/revision_20261110/local_validation')
    args=parser.parse_args()
    source=args.input_dir.resolve(); out=args.output_dir.resolve()
    if source==out or source in out.parents:
        raise ValueError('Output must not be inside the input prediction directory')
    out.mkdir(parents=True,exist_ok=True)
    reference_bytes=args.reference.read_bytes()
    reference=json.loads(reference_bytes)
    start=time.perf_counter(); rng=np.random.RandomState(0)
    input_hashes={}

    def ensemble(backbone,cohort):
        labels,scores=[],[]
        for seed in range(5):
            path=source/f'pred_{backbone}_seed{seed}_{cohort}.npz'
            with np.load(path,allow_pickle=False) as data:
                assert set(data.files)=={'ys','pos'}
                y,p=data['ys'],data['pos']
            assert y.ndim==p.ndim==1 and len(y)==len(p)
            assert np.isfinite(y).all() and np.isfinite(p).all()
            assert set(np.unique(y))=={0,1} and ((p>=0)&(p<=1)).all()
            labels.append(y); scores.append(p)
            input_hashes[path.name]=digest(path)
        assert all(np.array_equal(labels[0],y) for y in labels)
        # Preserve stored score dtype and the original five-seed mean.
        return labels[0].astype(int),np.mean(scores,axis=0)

    def ba_error(sensitivity,specificity,truth,n,prevalence):
        k=rng.binomial(n,prevalence,DRAWS_BA)
        present=(k>0)&(k<n)
        positive_n,negative_n=np.maximum(k,1),np.maximum(n-k,1)
        estimate=(rng.binomial(positive_n,sensitivity)/positive_n+
                  rng.binomial(negative_n,specificity)/negative_n)/2
        return np.where(present,np.abs(estimate-truth),np.inf)

    def auc_error(positive,negative,truth,n,prevalence):
        errors=[]
        for _ in range(DRAWS_AUC):
            k=rng.binomial(n,prevalence)
            if k==0 or k==n:
                errors.append(np.inf); continue
            p=rng.choice(positive,k,replace=True)
            q=np.sort(rng.choice(negative,n-k,replace=True))
            lo=np.searchsorted(q,p,'left'); hi=np.searchsorted(q,p,'right')
            errors.append(abs((lo+0.5*(hi-lo)).sum()/(k*(n-k))-truth))
        return np.array(errors)

    def required(error_function,tolerance):
        for n in GRID:
            if np.mean(error_function(n)<=tolerance)>=0.95:
                return int(n)
        return None

    result={'prevalences':PREVALENCES,'grid':GRID.tolist(),'rows':[]}
    curves={}
    # Order is deliberate: the shared RandomState matches the recorded run.
    for backbone,display in BACKBONES.items():
        for cohort in COHORTS:
            y,p=ensemble(backbone,cohort)
            positive,negative=p[y==1],p[y==0]
            se,sp=float(np.mean(positive>.5)),float(np.mean(negative<=.5))
            ba=(se+sp)/2; auc=float(roc_auc_score(y,p))
            row={'backbone':display,'cohort':cohort,'ba':round(ba,3),'auc':round(auc,3),'need':{}}
            for prevalence in PREVALENCES:
                fb=lambda n,prevalence=prevalence:ba_error(se,sp,ba,n,prevalence)
                fa=lambda n,prevalence=prevalence:auc_error(positive,negative,auc,n,prevalence)
                row['need'][str(prevalence)]={'ba_0.10':required(fb,.10),'ba_0.05':required(fb,.05),
                                              'auc_0.10':required(fa,.10),'auc_0.05':required(fa,.05)}
                curves.setdefault(('ba',display,prevalence),[]).append([float(np.percentile(fb(n),95)) for n in GRID])
                curves.setdefault(('auc',display,prevalence),[]).append([float(np.percentile(fa(n),95)) for n in GRID])
            result['rows'].append(row)
            print(display,cohort,'completed',flush=True)
    result['summary']={}
    for prevalence in PREVALENCES:
        for key in ('ba_0.10','ba_0.05','auc_0.10','auc_0.05'):
            values=[r['need'][str(prevalence)][key] for r in result['rows']]
            reached=[v for v in values if v is not None]
            result['summary'][f'{prevalence}|{key}']={
                'median_images':int(np.median(reached)) if reached else None,
                'range_images':[min(reached),max(reached)] if reached else None,
                'median_positives':round(float(np.median(reached))*prevalence,1) if reached else None,
                'reached':f'{len(reached)}/{len(values)}'}
    assert result==reference,'Recomputed sampling aggregate differs from the released reference'
    assert args.reference.read_bytes()==reference_bytes,'Reference changed during run'
    assert all(digest(source/name)==value for name,value in input_hashes.items())
    (out/'local_validation_cost.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(out/'local_validation_curves.npz',**{'|'.join(map(str,k)):np.asarray(v) for k,v in curves.items()})
    metadata={'schema_version':1,'aggregate_exact_match':True,'seed':0,'rng':'numpy.random.RandomState',
              'draws_ba':DRAWS_BA,'draws_auc':DRAWS_AUC,'cohorts':COHORTS,'backbones':BACKBONES,
              'grid':GRID.tolist(),'prevalences':PREVALENCES,
              'python':platform.python_version(),'numpy':np.__version__,'scikit_learn':sklearn.__version__,
              'elapsed_seconds':time.perf_counter()-start,'source_inputs_sha256':input_hashes,
              'script_sha256':digest(Path(__file__)),'reference_sha256':hashlib.sha256(reference_bytes).hexdigest(),
              'curve_shapes':{'|'.join(map(str,k)):list(np.asarray(v).shape) for k,v in curves.items()},
              'nonfinite_curve_counts':{'|'.join(map(str,k)):int((~np.isfinite(v)).sum()) for k,v in curves.items()},
              'nonfinite_interpretation':'Absent-class draws count as failures. Raw curve arrays retain inf/NaN; plot medians and IQRs use finite cohort quantiles.',
              'scope':'Public PDR/NPDR resampling simulation; no treatment-outcome clinical records, image identifiers or individual clinical predictions.'}
    (out/'cache_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print('Aggregate exact match; complete curves saved; seconds',round(metadata['elapsed_seconds'],1))


if __name__=='__main__':
    main()
