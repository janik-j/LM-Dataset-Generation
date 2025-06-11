"""
This script evaluates the performance of synthesis condition prediction models.

It processes raw text-based result files, parses ground truth and predicted
synthesis conditions (temperature, time), calculates regression metrics
(R², MAE, MSE), and generates summary tables and plots to compare model performance.
"""

import argparse
import glob
import json
import logging
import os
import re

import matplotlib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_PARAMETERS = [
    "Sintering Temperature",
    "Sintering Time",
    "Calcination Temperature",
    "Calcination Time",
]


def _is_prediction_valid(prediction_dict):
    """Check if a parsed prediction dictionary has valid numeric values."""
    if not isinstance(prediction_dict, dict):
        return False
    for param in EVAL_PARAMETERS:
        value = prediction_dict.get(param)
        if value is None:
            return False
        try:
            if np.isnan(float(value)):
                return False
        except (ValueError, TypeError):
            return False
    return True


def extract_predicted_values(raw_prediction):
    """Extract a dictionary from a raw prediction string."""
    match = re.search(r"\{[\s\S]*?\}", raw_prediction)
    if not match:
        return {}
    try:
        dict_str = match.group(0)
        # Remove comments before parsing
        cleaned_str = re.sub(r"#.*", "", dict_str)
        return json.loads(cleaned_str.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_section(section_content):
    """Parse a single entry section from a results file."""
    gt_match = re.search(
        r"Ground Truth Conditions:\s*(\{.*?\})", section_content, re.DOTALL
    )
    if not gt_match:
        return None, None
    try:
        gt_conditions = json.loads(gt_match.group(1))
    except json.JSONDecodeError:
        return None, None

    pred_raw_match = re.search(
        r"Predicted Conditions(?: \(Raw\))?:\s*(.*?)(?:\n-{5,}|$)",
        section_content,
        re.DOTALL,
    )
    pred_conditions = {}
    if pred_raw_match:
        pred_raw = pred_raw_match.group(1).strip()
        if pred_raw and pred_raw != "None":
            pred_conditions = extract_predicted_values(pred_raw)

    return gt_conditions, pred_conditions


def parse_results_file(filepath):
    """Parse a results file and return a list of processed entries."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except (FileNotFoundError, IOError) as e:
        logging.error("Error reading file %s: %s", filepath, e)
        return []

    sections = re.split(r"\n-{5,}\n", content.strip())
    unique_entries = {}

    for section in sections:
        if not section.strip():
            continue
        gt_conditions, pred_conditions = _parse_section(section)
        if gt_conditions is None:
            continue

        gt_key = json.dumps(gt_conditions, sort_keys=True)
        new_entry = {"ground_truth": gt_conditions, "prediction": pred_conditions}

        if gt_key not in unique_entries or _is_prediction_valid(pred_conditions):
            unique_entries[gt_key] = new_entry

    return list(unique_entries.values())


def _get_parameter_values(data, param):
    """Extract valid ground truth and predicted values for a single parameter."""
    gt_values, pred_values = [], []
    for item in data:
        gt_value = item.get("ground_truth", {}).get(param)
        pred_value = item.get("prediction", {}).get(param)
        if gt_value is not None and pred_value is not None:
            try:
                gt_num, pred_num = float(gt_value), float(pred_value)
                if not np.isnan(gt_num) and not np.isnan(pred_num):
                    gt_values.append(gt_num)
                    pred_values.append(pred_num)
            except (ValueError, TypeError):
                continue
    return gt_values, pred_values


def calculate_metrics(data):
    """Calculate regression metrics for each synthesis parameter."""
    results = {}
    for param in EVAL_PARAMETERS:
        gt_values, pred_values = _get_parameter_values(data, param)
        if len(gt_values) > 1:
            mse = mean_squared_error(gt_values, pred_values)
            results[param] = {
                "r2": r2_score(gt_values, pred_values),
                "mae": mean_absolute_error(gt_values, pred_values),
                "rmse": np.sqrt(mse),
                "valid_count": len(gt_values),
            }
        else:
            results[param] = {
                "r2": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "valid_count": 0,
            }
        results[param].update({"gt_values": gt_values, "pred_values": pred_values})
    return results


def plot_predictions(gt_values, pred_values, param_name, model_id, output_dir):
    """Generate and save a scatter plot of ground truth vs. predicted values."""
    if not gt_values or not pred_values:
        return
    try:
        plt.figure(figsize=(8, 8))
        plt.scatter(gt_values, pred_values, alpha=0.6, label="Predictions")
        min_val = min(min(gt_values), min(pred_values))
        max_val = max(max(gt_values), max(pred_values))
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal (y=x)")
        plt.title(f"{param_name}: True vs. Predicted\nModel: {model_id}")
        plt.xlabel(f"True {param_name}")
        plt.ylabel(f"Predicted {param_name}")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        safe_param_name = param_name.replace(" ", "_").replace("/", "_")
        plot_filename = os.path.join(
            output_dir, f"plot_{model_id}_{safe_param_name}.png"
        )
        plt.savefig(plot_filename, bbox_inches="tight")
        plt.close()
        logging.info("Plot saved to: %s", plot_filename)
    except Exception as e:
        logging.error("Error generating plot for %s (%s): %s", param_name, model_id, e)


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
            res = all_results[model_id].get(param, {})
            if res.get("valid_count", 0) > 0:
                table_data.append(
                    {
                        "identifier": model_id,
                        "r2_sort": res["r2"],
                        "r2_str": f"{res['r2']:.3f}",
                        "mae_str": f"{res['mae']:.2f}",
                        "rmse_str": f"{res['rmse']:.2f}",
                        "n_str": str(res["valid_count"]),
                    }
                )
        sorted_data = sorted(table_data, key=lambda x: x["r2_sort"], reverse=True)
        headers = ["Identifier", "R²", "MAE", "RMSE", "N"]
        rows = [
            [d["identifier"], d["r2_str"], d["mae_str"], d["rmse_str"], d["n_str"]]
            for d in sorted_data
        ]
        print(tabulate(rows, headers=headers, tablefmt="grid"))


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate synthesis condition predictions."
    )
    parser.add_argument(
        "--results_dir", default="results", help="Directory with results files."
    )
    args = parser.parse_args()

    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    filepaths = sorted(
        glob.glob(os.path.join(args.results_dir, "results_conditions_*.txt"))
    )

    if not filepaths:
        logging.warning("No files found in %s", args.results_dir)
        return

    all_results = {}
    for filepath in filepaths:
        model_id = os.path.basename(filepath).replace(".txt", "")
        logging.info("--- Evaluating: %s ---", model_id)
        data = parse_results_file(filepath)
        if data:
            metrics = calculate_metrics(data)
            all_results[model_id] = metrics
            for param in ["Sintering Temperature", "Calcination Temperature"]:
                if metrics.get(param, {}).get("valid_count", 0) > 0:
                    plot_predictions(
                        metrics[param]["gt_values"],
                        metrics[param]["pred_values"],
                        param,
                        model_id,
                        plots_dir,
                    )
    print_results_table(all_results)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
