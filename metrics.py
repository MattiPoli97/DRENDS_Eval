import numpy as np
 
def _affine_align_per_sequence(pred, gt, mask):
    """
    Fit s,t (least-squares) on all valid pixels across time to align pred to gt:
        minimize || s*pred + t - gt ||_2^2
    Returns aligned_pred, (s, t)
    """
    gt_v, pr_v= gt[mask], pred[mask]
    if gt_v.size == 0:
        return pred.copy(), (1.0, 0.0)  
 
    x = pr_v.astype(np.float64)
    y = gt_v.astype(np.float64)
 
    x_mean = x.mean()
    y_mean = y.mean()
    var_x  = np.mean((x - x_mean)**2)
 
    if var_x < 1e-12:
        s = 1.0
        t = y_mean - s * x_mean
    else:
        cov_xy = np.mean((x - x_mean) * (y - y_mean))
        s = cov_xy / var_x
        t = y_mean - s * x_mean
 
    aligned = s * pred + t
    return aligned, (float(s), float(t))
 
 
def mae(gt_v, pr_v):
    return float(np.mean(np.abs(gt_v - pr_v)))
 
def rmse(gt_v, pr_v):
    return float(np.sqrt(np.mean((gt_v - pr_v) ** 2)))
 
def rmse_log(gt_v, pr_v, eps=1e-6):
    return float(np.sqrt(np.mean((np.log(gt_v + eps) - np.log(pr_v + eps)) ** 2)))
 
def abs_rel(gt_v, pr_v, eps=1e-6):
    return float(np.mean(np.abs(gt_v - pr_v) / (gt_v + eps)))
 
def sq_rel(gt_v, pr_v, eps=1e-6):
    return float(np.mean(((gt_v - pr_v) ** 2) / (gt_v + eps)))
 
def delta_accuracies(gt_v, pr_v):
    """
    Returns a dict with a1, a2, a3 for thresholds 1.25^k, k=1,2,3
    """
    ratio = np.maximum(gt_v / np.maximum(pr_v, 1e-12), pr_v / np.maximum(gt_v, 1e-12))
    a1 = float(np.mean(ratio < 1.25))
    a2 = float(np.mean(ratio < (1.25 ** 2)))
    a3 = float(np.mean(ratio < (1.25 ** 3)))
    return {"a1": a1, "a2": a2, "a3": a3}
 
# ------------------------
# 1) Absolute-scale evaluation
# ------------------------
 
def absolute_scale_eval(GT_depth, Pred_depth, Masks, min_depth=1e-6):
    """
    Evaluate metrics on raw (un-aligned) depths.
    GT_depth, Pred_depth: (T, W, H)
    Returns dict of metrics.
    """
    gt_v, pr_v = GT_depth[Masks], Pred_depth[Masks]
    if gt_v.size == 0:
        return {"count": 0}

    out = {
        "count": int(gt_v.size),
        "L1_Error": mae(gt_v, pr_v),
        "RMSE": rmse(gt_v, pr_v),
        "RMSE_LOG": rmse_log(gt_v, pr_v),
        "ABS_REL": abs_rel(gt_v, pr_v),
        "SQ_REL": sq_rel(gt_v, pr_v),
    }
    out.update(delta_accuracies(gt_v, pr_v))
    return out
 
# ------------------------
# 2) Per-sequence aligned evaluation (s, t) shared across all frames
# ------------------------
 
def aligned_scale_eval(GT_depth, Pred_depth, Masks):
    """
    Affine-align predictions to GT over the entire sequence, then compute metrics.
    Returns dict of metrics + (s, t).
    """
    Pred_aligned, (s, t) = _affine_align_per_sequence(Pred_depth, GT_depth, Masks)
    gt_v, pr_v = Pred_aligned[Masks], GT_depth[Masks]
    if gt_v.size == 0:
        return {"count": 0, "s": s, "t": t}
 
    out = {
        "count": int(gt_v.size),
        "s": s,
        "t": t,
        "L1_Error": mae(gt_v, pr_v),
        "RMSE": rmse(gt_v, pr_v),
        "RMSE_LOG": rmse_log(gt_v, pr_v),
    }
    out.update(delta_accuracies(gt_v, pr_v))
    return out
 
# ------------------------
# 3) Temporal consistency evaluation: DTCE (L1 on differences-of-differences)
# ------------------------
 
def temporal_consistency_eval(GT_depth, Pred_depth, Masks, aligned=False):
    """
    Compute temporal consistency metrics on per-sequence aligned predictions.
    Returns dict with DTCE, ChangeCorr, ErrorFlicker and also (s, t).
    """
    if aligned:
        Pred_aligned, (s, t) = _affine_align_per_sequence(Pred_depth, GT_depth, Masks)
    else:
        Pred_aligned = Pred_depth.copy()
        s, t = 1.0, 0.0
    
    T = GT_depth.shape[0]
 
    d_gt = GT_depth[1:] - GT_depth[:-1]
    d_pr = Pred_aligned[1:] - Pred_aligned[:-1]
 
    valid_curr = Masks[ :-1]
    valid_next = Masks[1:  ]
    vmask = valid_curr & valid_next
 
    dtce = float(np.mean(np.abs((d_pr - d_gt)[vmask])))
 
    return {
        #"s": s,
        #"t": t,
        "DTCE": dtce,                 
    }