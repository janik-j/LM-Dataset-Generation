"""
This script evaluates and compares multiple synthesis condition prediction models,
including various Mixture of Experts (MoE) strategies.

It processes result files from individual models, finds a common set of data points,
and then calculates and compares the performance of different ensemble methods
(e.g., averaging, median) against the individual models.
"""

import argparse
import glob
import json
import logging
import os

import numpy as np
from tabulate import tabulate

# Reuse parsing and metric calculation from the standard evaluation script
from eval_conditions import (
    EVAL_PARAMETERS,
    calculate_metrics,
    parse_results_file,
    plot_predictions,
)


def _get_common_data(all_parsed_data, model_ids):
    """Find data points common across all models."""
    if not all_parsed_data or not model_ids:
        return {}, set()

    gt_to_preds = {}
    gt_key_sets = []

    for model_id in model_ids:
        model_keys = set()
        for item in all_parsed_data.get(model_id, []):
            try:
                gt_key = json.dumps(item["ground_truth"], sort_keys=True)
                model_keys.add(gt_key)
                if gt_key not in gt_to_preds:
                    gt_to_preds[gt_key] = {}
                gt_to_preds[gt_key][model_id] = item["prediction"]
            except TypeError:
                continue
        gt_key_sets.append(model_keys)

    common_keys = set.intersection(*gt_key_sets) if gt_key_sets else set()
    return gt_to_preds, common_keys


def _calculate_moe_predictions(gt_key, gt_to_preds, model_ids):
    """Calculate predictions for various MoE strategies for a single data point."""
    predictions_for_gt = gt_to_preds.get(gt_key, {})
    if len(predictions_for_gt) != len(model_ids):
        return None  # Skip if data is missing for any model

    try:
        ground_truth = json.loads(gt_key)
    except json.JSONDecodeError:
        return None

    moe_preds = {strategy: {} for strategy in ["Average", "Median"]}

    for param in EVAL_PARAMETERS:
        valid_preds = []
        for model_id in model_ids:
            pred_dict = predictions_for_gt.get(model_id, {})
            pred_value = pred_dict.get(param)
            if pred_value is not None:
                try:
                    num_value = float(pred_value)
                    if not np.isnan(num_value):
                        valid_preds.append(num_value)
                except (ValueError, TypeError):
                    continue

        if valid_preds:
            moe_preds["Average"][param] = np.mean(valid_preds)
            moe_preds["Median"][param] = np.median(valid_preds)
        else:
            moe_preds["Average"][param] = float("nan")
            moe_preds["Median"][param] = float("nan")

    return {
        strategy: {"ground_truth": ground_truth, "prediction": preds}
        for strategy, preds in moe_preds.items()
    }


def evaluate_moe_strategies(all_parsed_data, model_ids):
    """Evaluate different MoE strategies on the common data."""
    gt_to_preds, common_keys = _get_common_data(all_parsed_data, model_ids)
    if not common_keys:
        logging.warning(
            "No common data points found. Cannot calculate MoE performance."
        )
        return {}

    moe_data = {"MoE_Average": [], "MoE_Median": []}
    for gt_key in common_keys:
        entry_preds = _calculate_moe_predictions(gt_key, gt_to_preds, model_ids)
        if entry_preds:
            for strategy, entry in entry_preds.items():
                moe_data[f"MoE_{strategy}"].append(entry)

    moe_metrics = {}
    for strategy, data in moe_data.items():
        if data:
            moe_metrics[strategy] = calculate_metrics(data)
    return moe_metrics


def print_results_table(all_results):
    """Print a formatted table of results for each parameter."""
    if not all_results:
        print("No results to display.")
        return

    print("\n" + "=" * 80 + "\nDetailed Evaluation Results per Parameter\n" + "=" * 80)
    model_ids = list(all_results.keys())
    for param in EVAL_PARAMETERS:
        print(f"\n\n--- Results for: {param} ---")
        table_data = []
        for model_id in model_ids:
            res = all_results.get(model_id, {}).get(param, {})
            if res.get("valid_count", 0) > 0:
                table_data.append(
                    {
                        "Identifier": model_id,
                        "R²": f"{res['r2']:.4f}",
                        "MAE": f"{res['mae']:.3f}",
                        "MSE": f"{res['mse']:.3f}",
                        "N": str(res["valid_count"]),
                        "r2_sort": res["r2"],
                    }
                )
        sorted_data = sorted(table_data, key=lambda x: x["r2_sort"], reverse=True)
        for item in sorted_data:
            del item["r2_sort"]
        print(tabulate(sorted_data, headers="keys", tablefmt="grid"))


def main():
    """Main function to run the MoE evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate MoE condition prediction models."
    )
    parser.add_argument(
        "--results_dir", default="results", help="Directory with results files."
    )
    args = parser.parse_args()

    plots_dir = os.path.join(args.results_dir, "moe_plots")
    os.makedirs(plots_dir, exist_ok=True)
    filepaths = sorted(
        glob.glob(os.path.join(args.results_dir, "results_conditions_*.txt"))
    )

    if len(filepaths) < 2:
        logging.warning("Need at least 2 model result files to compute MoE results.")
        return

    all_metrics = {}
    all_parsed_data = {}
    model_ids = []

    for filepath in filepaths:
        model_id = os.path.basename(filepath).replace(".txt", "")
        logging.info("--- Evaluating: %s ---", model_id)
        data = parse_results_file(filepath)
        if data:
            all_parsed_data[model_id] = data
            all_metrics[model_id] = calculate_metrics(data)
            model_ids.append(model_id)

    if len(model_ids) >= 2:
        moe_metrics = evaluate_moe_strategies(all_parsed_data, model_ids)
        all_metrics.update(moe_metrics)

        for strategy, metrics in moe_metrics.items():
            for param in ["Sintering Temperature", "Calcination Temperature"]:
                if metrics.get(param, {}).get("valid_count", 0) > 0:
                    plot_predictions(
                        metrics[param]["gt_values"],
                        metrics[param]["pred_values"],
                        param,
                        strategy,
                        plots_dir,
                    )
    print_results_table(all_metrics)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
