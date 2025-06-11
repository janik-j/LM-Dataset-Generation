"""
This script evaluates the performance of retrosynthesis models by calculating
top-k accuracy from structured JSON result files.

It processes a directory of JSON files, each containing model predictions,
and calculates both exact and subset match accuracies for different k-values.
The results are then displayed in a summary table and saved to a JSON file.
"""

import argparse
import json
import os
import logging

from tabulate import tabulate


def parse_json_file(file_path):
    """Parse a JSON results file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            logging.warning("JSON data in %s is not a list.", file_path)
            return None
        return data
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        logging.error("Error parsing JSON file %s: %s", file_path, e)
        return None


def _get_ground_truth_set(entry):
    """Extract and process the ground truth from a result entry."""
    try:
        gt_precursors = entry.get("ground_truth")
        if not isinstance(gt_precursors, list):
            return None
        return {tuple(sorted(p)) if isinstance(p, list) else p for p in gt_precursors}
    except TypeError:
        return None


def _calculate_entry_ranks(entry, ground_truth_set):
    """Calculate the exact and subset match ranks for a single entry."""
    rank_exact, rank_subset = -1, -1
    found_exact, found_subset = False, False

    for idx, proposal in enumerate(entry.get("response", [])[:10]):
        if not isinstance(proposal, list):
            continue
        try:
            proposal_set = {
                tuple(sorted(p)) if isinstance(p, list) else p for p in proposal
            }
        except TypeError:
            continue

        current_rank = idx + 1
        if not found_exact and proposal_set == ground_truth_set:
            rank_exact = current_rank
            found_exact = True
        if not found_subset and ground_truth_set.issubset(proposal_set):
            rank_subset = current_rank
            found_subset = True

        if found_exact and found_subset:
            break

    return rank_exact, rank_subset


def calculate_accuracy(parsed_results):
    """Calculate top-k accuracy metrics from parsed results."""
    if not parsed_results:
        return None

    k_values = [1, 3, 5, 10]
    found_at_k = {k: 0 for k in k_values}
    found_at_k_exact = {k: 0 for k in k_values}
    valid_entries_count = 0

    for entry in parsed_results:
        ground_truth_set = _get_ground_truth_set(entry)
        if ground_truth_set is None:
            continue

        valid_entries_count += 1
        rank_exact, rank_subset = _calculate_entry_ranks(entry, ground_truth_set)

        for k in k_values:
            if rank_exact != -1 and rank_exact <= k:
                found_at_k_exact[k] += 1
            if rank_subset != -1 and rank_subset <= k:
                found_at_k[k] += 1

    if valid_entries_count == 0:
        return None

    return {
        "top_k": {k: found_at_k[k] / valid_entries_count for k in k_values},
        "top_k_exact": {k: found_at_k_exact[k] / valid_entries_count for k in k_values},
        "processed_count": valid_entries_count,
    }


def extract_model_name_from_json(filename):
    """Extract a model name from a result filename."""
    base_name = os.path.basename(filename)
    if base_name.startswith("results_") and base_name.endswith(".json"):
        parts = base_name[len("results_") : -len(".json")].split("_")
        return "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
    return "unknown_model"


def process_all_json_files(results_dir):
    """Process all JSON files in a directory and calculate accuracies."""
    all_results = []
    if not os.path.isdir(results_dir):
        logging.error("Results directory not found: %s", results_dir)
        return []

    json_files = [
        f
        for f in os.listdir(results_dir)
        if f.endswith(".json") and f.startswith("results_")
    ]
    logging.info("Found %d JSON files to process.", len(json_files))

    for json_file in json_files:
        file_path = os.path.join(results_dir, json_file)
        model_name = extract_model_name_from_json(json_file)
        parsed_data = parse_json_file(file_path)

        if parsed_data:
            accuracies = calculate_accuracy(parsed_data)
            if accuracies:
                all_results.append(_create_result_entry(model_name, accuracies))

    return all_results


def _create_result_entry(model_name, accuracies):
    """Create a dictionary entry for the results summary."""
    result_entry = {"Model": model_name, "Targets": accuracies["processed_count"]}
    for k, acc in accuracies.get("top_k", {}).items():
        result_entry[f"Top-{k} Subset"] = f"{acc:.4f}"
    for k, acc in accuracies.get("top_k_exact", {}).items():
        result_entry[f"Top-{k} Exact"] = f"{acc:.4f}"
    return result_entry


def save_comparison_to_json(results_list, output_dir, filename):
    """Save the comparison results to a JSON file."""
    if not results_list:
        return

    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=4, ensure_ascii=False)
        logging.info("Successfully saved comparison summary to: %s", output_path)
    except (IOError, TypeError) as e:
        logging.error("Error writing JSON comparison file: %s", e)


def display_comparison_results(results_list):
    """Display the comparison results in a formatted table."""
    if not results_list:
        return

    print("\n===== MODEL COMPARISON SUMMARY (Exact Matches Only) =====")
    try:
        results_list.sort(key=lambda x: float(x.get("Top-1 Exact", 0.0)), reverse=True)
    except (ValueError, TypeError):
        logging.warning("Could not sort results numerically.")

    headers = ["Model", "Targets"] + [f"Top-{k} Exact" for k in [1, 3, 5, 10]]
    display_data = [{h: r.get(h, "N/A") for h in headers} for r in results_list]
    print(tabulate(display_data, headers="keys", tablefmt="grid"))


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Calculate accuracy from results JSON files."
    )
    parser.add_argument(
        "--results_dir",
        default="results_jsons",
        help="Directory with input JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        default="json_results",
        help="Directory to save the summary file.",
    )
    parser.add_argument(
        "--summary_filename",
        default="comparison_summary.json",
        help="Name for the output summary file.",
    )
    args = parser.parse_args()

    comparison_results = process_all_json_files(args.results_dir)
    save_comparison_to_json(comparison_results, args.output_dir, args.summary_filename)
    display_comparison_results(comparison_results)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
