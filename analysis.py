#!/usr/bin/env python3

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path("../ExVivo_results_new")
LIGHT_LEVELS = ["Ext", "Med", "High"]

def safe_mean_std_entry(d):
    """Return (mean, std) as floats, converting None to np.nan."""
    m = d.get("mean") if isinstance(d, dict) else None
    s = d.get("std") if isinstance(d, dict) else None
    return (np.nan if m is None else m, np.nan if s is None else s)


def load_metrics(root_dir):
    """
    Walk root_dir and load metrics_summary.json files.
    Returns:
      - batch_names: sorted list of batch folder names
      - metrics: dict[batch][model] -> metrics dict (as loaded from JSON)
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    batch_names = []
    metrics_per_batch = {}

    for batch in sorted(p.name for p in root_dir.iterdir() if p.is_dir()):
        if not batch.startswith("Seq"):
            continue
        batch_path = root_dir / batch
        batch_names.append(batch)
        metrics_per_batch[batch] = {}

        for model_dir in sorted(p for p in batch_path.iterdir() if p.is_dir()):
            metrics_file = model_dir / "metrics_summary.json"
            if not metrics_file.is_file():
                continue
            try:
                with metrics_file.open("r") as fh:
                    metrics = json.load(fh)
            except Exception:
                # skip files that cannot be parsed
                continue
            metrics_per_batch[batch][model_dir.name] = metrics

    return batch_names, metrics_per_batch


def collect_top_and_metric_names(metrics_per_batch):
    """Collect all top-level keys and per-top-key metric names across all models/batches."""
    top_keys = set()
    for batch_metrics in metrics_per_batch.values():
        for model_metrics in batch_metrics.values():
            top_keys.update(model_metrics.keys())

    metric_names_by_top = {}
    for top_key in sorted(top_keys):
        metric_names = set()
        for batch_metrics in metrics_per_batch.values():
            for model_metrics in batch_metrics.values():
                if top_key in model_metrics and isinstance(model_metrics[top_key], dict):
                    metric_names.update(model_metrics[top_key].keys())
        metric_names_by_top[top_key] = sorted(metric_names)
    return sorted(top_keys), metric_names_by_top


def get_all_models(metrics_per_batch):
    """Return sorted list of unique model names across all batches."""
    return sorted({m for batch in metrics_per_batch.values() for m in batch.keys()})


def plot_per_batch_and_by_light(batch_names, metrics_per_batch, saving_folder):
    """
    For each top_key and metric, create:
      - per-batch plot with models side-by-side (points with error bars)
      - aggregated plot by light level (Ext/Med/High) averaging across batches
    """
    top_keys, metric_names_by_top = collect_top_and_metric_names(metrics_per_batch)
    all_models = get_all_models(metrics_per_batch)
    x_batches = np.arange(len(batch_names))

    for top_key in top_keys:
        for metric in metric_names_by_top.get(top_key, []):

            # Per-batch plot (models side-by-side)
            plt.figure(figsize=(10, 6))
            n_models = max(1, len(all_models))
            total_width = 0.8
            bar_width = total_width / n_models

            for i, model in enumerate(all_models):
                means, stds = [], []
                for batch in batch_names:
                    entry = metrics_per_batch[batch].get(model, {}).get(top_key, {}).get(metric, {})
                    mean_v, std_v = safe_mean_std_entry(entry)
                    means.append(mean_v)
                    stds.append(std_v)
                x_shift = x_batches - total_width / 2 + (i + 0.5) * bar_width
                plt.errorbar(x_shift, means, yerr=stds, fmt="o", linestyle="None", capsize=3, label=model)

            plt.title(f"{top_key} - {metric} (mean ± std) across batches")
            plt.xlabel("Batch")
            plt.ylabel(metric)
            plt.xticks(x_batches, batch_names, rotation=45)
            plt.tight_layout()
            out_file = saving_folder / f"{top_key}_{metric}.png"
            plt.savefig(out_file)
            plt.close()

            # Aggregated by light level (all models plotted separately)
            plt.figure(figsize=(8, 5))
            x_lights = np.arange(len(LIGHT_LEVELS))
            total_width = 0.7
            model_width = total_width / max(1, len(all_models))

            for mi, model in enumerate(all_models):
                light_means = {L: [] for L in LIGHT_LEVELS}
                light_stds = {L: [] for L in LIGHT_LEVELS}

                for batch in batch_names:
                    entry = metrics_per_batch[batch].get(model, {}).get(top_key, {}).get(metric, {})
                    mean_v, std_v = safe_mean_std_entry(entry)
                    for L in LIGHT_LEVELS:
                        if L in batch:
                            light_means[L].append(mean_v)
                            light_stds[L].append(std_v)
                            break

                agg_means = [np.nanmean(light_means[L]) if light_means[L] else np.nan for L in LIGHT_LEVELS]
                agg_stds = [np.nanmean(light_stds[L]) if light_stds[L] else np.nan for L in LIGHT_LEVELS]
                x_shift = x_lights - total_width / 2 + (mi + 0.5) * model_width
                plt.errorbar(x_shift, agg_means, yerr=agg_stds, fmt="o", linestyle="None", capsize=3, label=model)

            plt.xticks(x_lights, LIGHT_LEVELS)
            plt.xlabel("Light level")
            plt.ylabel(metric)
            plt.title(f"{top_key} - {metric} (mean ± std) by light level")
            plt.tight_layout()
            out_file = saving_folder / f"{top_key}_{metric}_by_light.png"
            plt.savefig(out_file)
            plt.close()


def aggregate_channel_metrics(batch_names, metrics_per_batch, channel_key):
    """
    Aggregate metrics for channel_key (e.g., 'left_raw', 'right_raw', 'left_aligned', 'right_aligned').
    Returns dict[model][metric_name] -> {'mean': [..], 'std': [..]}
    """
    agg = {}
    for batch in batch_names:
        for model, mdict in metrics_per_batch[batch].items():
            channel_metrics = mdict.get(channel_key, {})
            if model not in agg:
                agg[model] = {}
            for metric_name, metric_values in channel_metrics.items():
                if metric_name not in agg[model]:
                    agg[model][metric_name] = {"mean": [], "std": []}
                mean_v, std_v = safe_mean_std_entry(metric_values)
                agg[model][metric_name]["mean"].append(mean_v)
                agg[model][metric_name]["std"].append(std_v)
    return agg


def format_table_from_agg(agg):
    """
    Convert aggregated dict into table-ready dict mapping model -> metric -> "mean ± std"
    Uses np.nanmean across collected values.
    """
    table = {}
    metric_names = set()
    for model, metrics_dict in sorted(agg.items()):
        table[model] = {}
        for metric_name, values in metrics_dict.items():
            mean_val = np.nanmean(values["mean"])
            std_val = np.nanmean(values["std"])
            table[model][metric_name] = f"{mean_val:.3f} ± {std_val:.3f}"
            metric_names.add(metric_name)
    return table, sorted(metric_names)


def save_summary_tables(saving_folder, batch_names, metrics_per_batch):
    """Produce CSV summary tables for raw and aligned, left and right channels."""
    # aggregate for each channel
    left_raw = aggregate_channel_metrics(batch_names, metrics_per_batch, "left_raw")
    right_raw = aggregate_channel_metrics(batch_names, metrics_per_batch, "right_raw")
    left_al = aggregate_channel_metrics(batch_names, metrics_per_batch, "left_aligned")
    right_al = aggregate_channel_metrics(batch_names, metrics_per_batch, "right_aligned")

    for data, filename in (
        (left_raw, "left_raw_metrics_summary_by_model.csv"),
        (right_raw, "right_raw_metrics_summary_by_model.csv"),
        (left_al, "left_aligned_metrics_summary_by_model.csv"),
        (right_al, "right_aligned_metrics_summary_by_model.csv"),
    ):
        table_dict, metric_names = format_table_from_agg(data)
        if not table_dict:
            # nothing to save
            continue
        df = pd.DataFrame.from_dict(table_dict, orient="index")
        # Ensure consistent column order
        df = df[[c for c in sorted(metric_names) if c in df.columns]]
        out_path = saving_folder / filename
        df.to_csv(out_path)


def main():
    saving_folder = ROOT_DIR / "plots"
    saving_folder.mkdir(parents=True, exist_ok=True)

    batch_names, metrics_per_batch = load_metrics(ROOT_DIR)
    if not batch_names:
        print("No batches found. Exiting.")
        return

    plot_per_batch_and_by_light(batch_names, metrics_per_batch, saving_folder)
    save_summary_tables(saving_folder, batch_names, metrics_per_batch)
    print(f"Saved plots and tables to: {saving_folder}")


if __name__ == "__main__":
    main()
