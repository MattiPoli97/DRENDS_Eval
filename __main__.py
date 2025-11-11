#!/usr/bin/env python3
import argparse
from email import parser
from pathlib import Path
import cv2
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile
from imageio.v2 import imread
import matplotlib.pyplot as plt

from datasets import *          
from metrics import absolute_scale_eval, aligned_scale_eval, temporal_consistency_eval   
from models.interfaces import ( 
    BaseInterface,
    DAMv2, DAMv2_Metric,
    MiDaS, ZoeDepthInterface, DepthProInterface, MonoDepth2,
)

# ------------------------ helpers ------------------------
def collate_paths(batch):
    """Keep paths as strings; batch_size is expected to be 1."""
    out = {k: [] for k in batch[0].keys()}
    for item in batch:
        for k, v in item.items():
            out[k].append(v)
    return out

def normalize_for_save(m):
    """Min-max -> 8-bit PNG (visualization only)."""
    m = np.asarray(m).astype(np.float32)
    vmin = np.nanmin(m)
    vmax = np.nanmax(m)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(m, dtype=np.uint8)
    out = (np.clip((m - vmin) / (vmax - vmin), 0, 1) * 255.0).astype(np.uint8)
    return out

def make_depth_model(name: str, img_size, **kwargs) -> BaseInterface:
    """
    name (case-insensitive):
      - damv2, damv2_metric, midas, zoedepth, depthpro, monovit, monodepth2
    kwargs are forwarded to the model's __init__.
    """
    name = name.lower()
    if name == "damv2":
        return DAMv2(img_size, **kwargs)
    if name == "damv2_metric":
        return DAMv2_Metric(img_size, **kwargs)
    if name == "midas":
        return MiDaS(img_size, **kwargs)
    if name == "zoedepth":
        return ZoeDepthInterface(img_size, **kwargs)
    if name == "depthpro":
        return DepthProInterface(img_size, **kwargs)
    if name == "monodepth2":
        return MonoDepth2(img_size, **kwargs)
    raise ValueError(f"Unknown model name: {name}")

def model_infer(model: BaseInterface, image_path: str) -> np.ndarray:
    """
    Run the model on a single image path and return depth (H,W) float32.
    All our model interfaces implement __call__(path)->np.ndarray.
    """
    depth = model(image_path)
    if not isinstance(depth, np.ndarray):
        depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    assert depth.ndim == 2, f"Expected (H,W) depth, got shape {depth.shape}"
    return depth.astype(np.float32)

def depth_to_metric(M, d_min, d_max, mask=None, use_percentiles=True, q=(2, 98), eps=1e-6):
    """
    Convert MiDaS relative inverse depth M to metric depth in meters,
    given known real-depth range [d_min, d_max].

    - M: np.ndarray (H, W), MiDaS output (relative inverse depth; larger=closer)
    - mask: optional boolean mask for valid pixels (same HxW)
    - use_percentiles: use robust min/max from percentiles to ignore outliers
    - q: lower/upper percentiles if use_percentiles=True
    """
    M = np.asarray(M, dtype=np.float64)
    if mask is None:
        valid = np.isfinite(M)
    else:
        valid = (mask.astype(bool)) & np.isfinite(M)

    if not np.any(valid):
        return np.full_like(M, np.nan, dtype=np.float32), (1.0, 0.0)

    vals = M[valid]
    if use_percentiles:
        M_min, M_max = np.percentile(vals, [q[0], q[1]])
    else:
        M_min, M_max = np.min(vals), np.max(vals)

    if not np.isfinite(M_min) or not np.isfinite(M_max) or M_max <= M_min + 1e-12:
        # degenerate range: return mid-depth
        return np.full_like(M, (d_min + d_max) * 0.5, dtype=np.float32), (1.0, 0.0)

    inv_d_min = 1.0 / float(d_min)
    inv_d_max = 1.0 / float(d_max)

    s = (inv_d_min - inv_d_max) / (M_max - M_min)
    t = inv_d_max - s * M_min

    denom = s * M + t
    denom = np.maximum(denom, eps)  # keep positive
    D = 1.0 / denom
    # clamp to the known physical range (optional but nice)
    D = np.clip(D, d_min, d_max).astype(np.float32)
    return D, (float(s), float(t))

def _load_gt_and_mask(gt_path, mask_path):
    # GT: TIFF (float). Mask: png/jpg/bw/rgb -> boolean
    gt = tifffile.imread(gt_path).astype(np.float64)
    m = imread(mask_path)
    if m.ndim == 3:  # RGB mask: any nonzero -> True
        m = np.any(m != 0, axis=2)
    m = (m != 0)
    return gt, m

def _valid_mask_for_alignment(gt, pred, user_mask):
    eps = 1e-6
    v = user_mask & np.isfinite(gt) & np.isfinite(pred) & (gt > eps) & (pred > eps)
    return v

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate depth models on RoboLab3D (masked metrics).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--model", type=str, default="monodepth2",
                        help="Model: damv2 | damv2_metric | midas | zoedepth | depthpro | monodepth2 | metric3d | unidepth")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save predictions & summary")
    parser.add_argument("--batch_size", type=int, default=1, help="Keep 1 (interfaces infer per-path)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--normalize_metrics", action="store_true",
                        help="Min-max normalize GT and Pred over the valid mask before metrics")

    parser.add_argument("--midas_type", type=str, default="DPT_Hybrid")
    parser.add_argument("--damv2_size", type=str, default="small")
    parser.add_argument("--damv2m_size", type=str, default="Base")
    parser.add_argument("--damv2m_domain", type=str, default="Indoor")   
    parser.add_argument("--zoe_name", type=str, default="zoedepth_nk")
    parser.add_argument("--monodepth2_ckpt", type=str, default="mono_640x192")

    parser.add_argument("--damv2m_max_depth", type=float, default=5.0)
    parser.add_argument("--zoe_max_depth", type=float, default=0.6)
    parser.add_argument("--headless", action="store_true", help="Visualization")
    parser.add_argument("--debug", action="store_true", help="Debug prints")
    parser.add_argument("--store_pngs", action="store_true", help="Store visual PNGs of predictions")
    
    return parser.parse_args()

def build_dataloader(args):
    dataset = RoboLab3D(args.data_path)
    data_list = dataset.get_images()
    img_size = dataset.__get_size__()

    if args.batch_size != 1:
        print("Note: For these interfaces, batch_size>1 gives no speedup.")

    dataloader = DataLoader(
        data_list,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_paths,
    )
    return dataloader, img_size

def build_model(args, img_size):
    if args.model.lower() == "midas":
        model = make_depth_model("midas", img_size, model_type=args.midas_type)
    elif args.model.lower() == "damv2":
        model = make_depth_model("damv2", img_size, model_size=args.damv2_size)
    elif args.model.lower() == "damv2_metric":
        model = make_depth_model("damv2_metric", img_size,
                                 model_size=args.damv2m_size, domain=args.damv2m_domain, max_depth=args.damv2m_max_depth)
    elif args.model.lower() == "zoedepth":
        model = make_depth_model("zoedepth", img_size, max_depth=args.zoe_max_depth)
    elif args.model.lower() == "depthpro":
        model = make_depth_model("depthpro", img_size)
    elif args.model.lower() == "monodepth2":
        model = make_depth_model("monodepth2", img_size, model_type=args.monodepth2_ckpt)
    else:
        raise ValueError(f"Unknown --model {args.model}")
    return model

# ------------------------ main ------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.debug:
        print(f"Using device: {device} | Model: {args.model} | Data: {args.data_path}")

    dataloader, img_size = build_dataloader(args)
    if args.debug:
        print(f"Dataset size: {len(dataloader.dataset)} images | Image size: {img_size}")

    model = build_model(args, img_size)

    # Output dir
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics accumulator
    keys = ["L1_Error","ABS_REL","SQ_REL","RMSE","RMSE_LOG","a1","a2","a3"]
    keys_aligned = ["L1_Error","RMSE","RMSE_LOG","a1","a2","a3"]
    acc_raw    = {k: [] for k in keys}
    acc_aligned = {k: [] for k in keys_aligned}


    for batch in tqdm(dataloader, desc=f"Evaluating {args.model}"):
        left_path   = batch.get("left_image",   [None])[0]
        right_path  = batch.get("right_image",  [None])[0]
        gt_left     = batch.get("gt_left_image",[None])[0]
        gt_right    = batch.get("gt_right_image",[None])[0]
        mask_left   = batch.get("mask_left",    [None])[0]
        mask_right  = batch.get("mask_right",   [None])[0]
        
        def process_one(img_path, gt_path, mask_path, suffix, debug):
            if img_path is None or gt_path is None or mask_path is None:
                return            
            pred = model_infer(model, img_path)  
            pred = pred.astype(np.float64) * 1000.0  # convert to mm
            gt, mask = _load_gt_and_mask(gt_path, mask_path)
            valid = _valid_mask_for_alignment(gt, pred, mask)
            
            gt_valid = gt[valid & (gt > 0)].astype(np.float64)  
            gt_min = (gt_valid.min()) if gt_valid.size > 0 else 0.0
            gt_max = (gt_valid.max()) if gt_valid.size > 0 else 0.0

            if args.model== "midas" or args.model == "damv2":
                gt_min = 100
                gt_max = 600
                pred, _ = depth_to_metric(pred, gt_min, gt_max, mask=mask)
       
            # 1) RAW metrics 
            m_raw = absolute_scale_eval(gt, pred, mask, min_depth=1e-6)
            for k in keys: acc_raw[k].append(m_raw[k])

            # 2) Aligned metrics
            m_met = aligned_scale_eval(gt, pred, mask)
            for k in keys_aligned: acc_aligned[k].append(m_met[k])

            # 3) Temporl consistency 
            dtce_aligned = temporal_consistency_eval(gt, pred, mask, aligned=True)
            dtce = temporal_consistency_eval(gt, pred, mask, aligned=False)
           
            stem = Path(img_path).stem
            if args.store_pngs:
                cv2.imwrite(str(out_dir / f"{stem}_{suffix}.png"),     normalize_for_save(pred))
  
            vmin = np.nanmin(gt[valid])
            vmax = np.nanmax(gt[valid])

            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axs[0].imshow(np.where(valid, gt, np.nan), vmin=vmin, vmax=vmax, cmap='viridis')
            axs[0].set_title('GT (valid pixels)')
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
            im1 = axs[1].imshow(np.where(valid, pred, np.nan), vmin=vmin, vmax=vmax, cmap='viridis')
            axs[1].set_title('Prediction (valid pixels)')
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
            plt.tight_layout()
            if args.store_pngs:
                plt.savefig(str(out_dir / f"{stem}_gt_pred_metric_{suffix}.png"))
            if not args.headless:
                plt.show()
            
            plt.close(fig)
            
            return m_raw, m_met, dtce, dtce_aligned

        m_raw_left, m_met_left, dtce_left, dtce_aligned_left = process_one(left_path,  gt_left,  mask_left,  "L", args.debug)
        m_raw_right, m_met_right, dtce_right, dtce_aligned_right = process_one(right_path, gt_right, mask_right, "R", args.debug)

        per_image_metrics = {
            "left_image": str(left_path),
            "right_image": str(right_path),
            "left_raw": {k: float(m_raw_left[k]) for k in keys},
            "left_aligned": {k: float(m_met_left[k]) for k in keys_aligned},
            "right_raw": {k: float(m_raw_right[k]) for k in keys},
            "right_aligned": {k: float(m_met_right[k]) for k in keys_aligned},
            "temporal_left": dtce_left["DTCE"],
            "temporal_right": dtce_right["DTCE"],
            "temporal_aligned_left": dtce_aligned_left["DTCE"],
            "temporal_aligned_right": dtce_aligned_right["DTCE"]
        }
        per_image_json_path = out_dir / "single_image_metrics.json"
        if per_image_json_path.exists():
            with open(per_image_json_path, "r") as f:
                all_metrics = json.load(f)
            if not isinstance(all_metrics, list):
                all_metrics = [all_metrics]
        else:
            all_metrics = []
        all_metrics.append(per_image_metrics)
        with open(per_image_json_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

    per_image_json_path = out_dir / "single_image_metrics.json"
    if per_image_json_path.exists():
        with open(per_image_json_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []

    def compute_stats(metric_list, keys):
        stats = {}
        for k in keys:
            vals = [img_metrics[k] for img_metrics in metric_list if k in img_metrics]
            arr = np.array(vals, dtype=np.float32)
            stats[k] = {
                "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
                "std": float(np.nanstd(arr)) if arr.size else float("nan")
            }
        return stats

    left_raw_list = [img["left_raw"] for img in all_metrics if "left_raw" in img]
    for i, img in enumerate(all_metrics):
        if "temporal_left" in img and i < len(left_raw_list):
            left_raw_list[i]["temporal_left"] = img["temporal_left"]
    right_raw_list = [img["right_raw"] for img in all_metrics if "right_raw" in img]
    for i, img in enumerate(all_metrics):
        if "temporal_right" in img and i < len(right_raw_list):
            right_raw_list[i]["temporal_right"] = img["temporal_right"]
    left_aligned_list = [img["left_aligned"] for img in all_metrics if "left_aligned" in img]
    for i, img in enumerate(all_metrics):
        if "temporal_aligned_left" in img and i < len(left_aligned_list):
            left_aligned_list[i]["temporal_aligned_left"] = img["temporal_aligned_left"]
    right_aligned_list = [img["right_aligned"] for img in all_metrics if "right_aligned" in img]
    for i, img in enumerate(all_metrics):
        if "temporal_aligned_right" in img and i < len(right_aligned_list):
            right_aligned_list[i]["temporal_aligned_right"] = img["temporal_aligned_right"]

    keys_left = keys.copy()
    keys_right = keys.copy()
    keys_aligned_left = keys_aligned.copy()
    keys_aligned_right = keys_aligned.copy()
    if "temporal_left" not in keys:
        keys_left.append("temporal_left")
    if "temporal_right" not in keys:
        keys_right.append("temporal_right")
    if "temporal_aligned_left" not in keys_aligned:
        keys_aligned_left.append("temporal_aligned_left")
    if "temporal_aligned_right" not in keys_aligned:
        keys_aligned_right.append("temporal_aligned_right")

    summary = {
        "left_raw": compute_stats(left_raw_list, keys_left),
        "right_raw": compute_stats(right_raw_list, keys_right),
        "left_aligned": compute_stats(left_aligned_list, keys_aligned_left),
        "right_aligned": compute_stats(right_aligned_list, keys_aligned_right)
    }

    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print("Summary (mean/std):", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
