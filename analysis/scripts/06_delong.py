# -*- coding: utf-8 -*-
"""
06_delong.py — DeLong 检验:比较 image-only vs image+clinical+syndrome 的 AUC 是否显著不同
============================================================
对每折做配对 DeLong(同一验证集样本),报告 AUC 差异与 p 值;并做合并(pooled)检验。
依赖: results/multimodal_antivegf/valpred_{mode}_fold{k}.npz
诚实原则:如 p>=0.05,论文不得宣称"显著提升",只能说"最佳综合性能/最高敏感度"。
"""
import os, sys, glob
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 可用命令行参数指定结果目录(默认concat融合;传 multimodal_gated 测gated融合)
_subdir = sys.argv[1] if len(sys.argv) > 1 else "multimodal_antivegf"
D = os.path.join(ROOT, "results", _subdir)


# ---- DeLong 实现(Sun & Xu 2014 快速版)----
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def fast_delong(preds_sorted, m):
    # preds_sorted: [2, n] 行=两个模型;前 m 列为正类
    n = preds_sorted.shape[1] - m; k = preds_sorted.shape[0]
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]
    tx = np.array([compute_midrank(pos[r]) for r in range(k)])
    ty = np.array([compute_midrank(neg[r]) for r in range(k)])
    tz = np.array([compute_midrank(preds_sorted[r]) for r in range(k)])
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y, p1, p2):
    order = (-np.array(y)).argsort()  # 正类(1)在前
    label = np.array(y)[order]; m = int(label.sum())
    preds = np.vstack([np.array(p1)[order], np.array(p2)[order]])
    aucs, cov = fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0: var = 1e-12
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, p


def main():
    mA, mB = "image", "image_clinical_syndrome"
    print(f"DeLong: A={mA}  vs  B={mB}\n")
    allp1, allp2, ally = [], [], []
    diffs = []
    for k in range(5):
        fa = os.path.join(D, f"valpred_{mA}_fold{k}.npz")
        fb = os.path.join(D, f"valpred_{mB}_fold{k}.npz")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            print(f"fold{k}: 缺概率文件,跳过"); continue
        da, db = np.load(fa), np.load(fb)
        ys = da["ys"]
        a1, a2, z, p = delong_test(ys, da["pos"], db["pos"])
        diffs.append(a2 - a1)
        sig = "*" if p < 0.05 else "ns"
        print(f"fold{k}: AUC_A={a1:.3f}  AUC_B={a2:.3f}  diff={a2-a1:+.3f}  p={p:.3f} [{sig}]")
        allp1.append(da["pos"]); allp2.append(db["pos"]); ally.append(ys)

    if ally:
        y = np.concatenate(ally); p1 = np.concatenate(allp1); p2 = np.concatenate(allp2)
        a1, a2, z, p = delong_test(y, p1, p2)
        print(f"\n[POOLED] AUC_A={a1:.3f}  AUC_B={a2:.3f}  diff={a2-a1:+.3f}  p={p:.4f}")
        print(f"平均每折 AUC 差异: {np.mean(diffs):+.3f} ± {np.std(diffs):.3f}")
        # 同时判断显著性 + 方向(diff = B - A;正=B更好,负=B更差)
        diff = a2 - a1
        if p >= 0.05:
            verdict = "B(加证型)与A(纯影像)无统计差异(p>=0.05)→ 证型未带来显著提升,不可称'显著提升AUC'"
        elif diff > 0:
            verdict = "B(加证型)显著优于A(p<0.05 且 diff>0)→ 证型显著提升 AUC"
        else:
            verdict = "⚠️ B(加证型)显著差于A(p<0.05 且 diff<0)→ 加证型反而显著降低 AUC(很可能过拟合)"
        print("\n结论:", verdict)


if __name__ == "__main__":
    main()
