"""
This script converts text-based model prediction results into a structured JSON format.

It parses raw .txt files from a specified directory, extracts target materials,
ground truth precursors, and model responses, and then saves the structured data
as a .json file. It includes model-specific parsing logic and deduplicates
entries based on the target material, prioritizing more complete records.
"""

import argparse
import ast
import json
import os
import re
import logging

from pymatgen.core import Composition

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def standardize_formula(formula):
    """Standardize a chemical formula using pymatgen."""
    if not isinstance(formula, str) or not formula or formula == "None":
        return formula
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return formula


def _parse_llama_response(response_section):
    """Parse a response specifically from a Llama model."""
    # 1. Try to find Python code blocks
    python_blocks = re.findall(r"```(?:python)?\n(.*?)```", response_section, re.DOTALL)
    if python_blocks:
        for block in python_blocks:
            block_str = block.strip()
            if block_str.startswith("[") and block_str.endswith("]"):
                try:
                    result = ast.literal_eval(block_str)
                    if (
                        isinstance(result, list)
                        and result
                        and all(isinstance(item, list) for item in result)
                    ):
                        return result
                except (SyntaxError, ValueError):
                    pass

    # 2. If no valid response yet, try to find any top-level list
    list_pattern = r"\[\s*\[.*?\]\s*(?:,\s*\[.*?\]\s*)*\]"
    list_matches = re.findall(list_pattern, response_section, re.DOTALL)
    for match in list_matches:
        try:
            result = ast.literal_eval(match)
            if (
                isinstance(result, list)
                and result
                and all(isinstance(item, list) for item in result)
            ):
                return result
        except (SyntaxError, ValueError):
            pass

    # 3. If still no valid response, try to extract line by line
    precursor_lists = []
    for line in response_section.split("\n"):
        line = line.strip()
        if line.startswith("[") and line.endswith("]") and "," in line:
            try:
                result = ast.literal_eval(line)
                if isinstance(result, list):
                    precursor_lists.append(result)
            except (SyntaxError, ValueError):
                sanitized = re.sub(r"[^\[\],\'\"a-zA-Z0-9\(\)\._ -]", "", line)
                try:
                    result = ast.literal_eval(sanitized)
                    if isinstance(result, list):
                        precursor_lists.append(result)
                except (SyntaxError, ValueError):
                    pass
    if precursor_lists:
        return precursor_lists

    return []


def _parse_openai_qwen_response(response_section):
    """Parse a response from OpenAI or Qwen models."""
    match = re.search(r"^\[.*\]", response_section, re.MULTILINE | re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            pass
    return []


def _parse_generic_response(response_section):
    """A generic fallback parser for unknown model types."""
    # This function can be expanded with more generic strategies.
    # First, try python code blocks
    python_blocks = re.findall(r"```(?:python)?\n(.*?)```", response_section, re.DOTALL)
    if python_blocks:
        for block in python_blocks:
            try:
                result = ast.literal_eval(block.strip())
                if isinstance(result, list) and all(
                    isinstance(item, list) for item in result
                ):
                    return result
            except (SyntaxError, ValueError):
                continue

    # If no code blocks, look for list patterns directly
    return _parse_openai_qwen_response(response_section)


def get_response_parser(model_type):
    """Return the appropriate response parsing function based on model type."""
    if model_type == "llama":
        return _parse_llama_response
    if model_type in ["openai", "qwen", "gemini", "deepseek"]:
        return _parse_openai_qwen_response
    return _parse_generic_response


def parse_entry(entry_content, response_parser):
    """Parse a single entry from a results file."""
    target_match = re.search(r"Target material: (.+)", entry_content)
    if not target_match:
        return None

    target = standardize_formula(target_match.group(1).strip())
    parsed_entry = {"target": target, "ground_truth": [], "response": []}

    gt_match = re.search(
        r"Ground truth precursors: (.*?)(?:\nResponse:|$)", entry_content, re.DOTALL
    )
    if gt_match:
        try:
            gt_list = ast.literal_eval(gt_match.group(1).strip())
            if isinstance(gt_list, list):
                parsed_entry["ground_truth"] = [standardize_formula(f) for f in gt_list]
        except (SyntaxError, ValueError):
            pass

    response_match = re.search(r"Response:(.*)", entry_content, re.DOTALL)
    if response_match:
        response_section = response_match.group(1).strip()
        response_lists = response_parser(response_section)

        if not response_lists:
            response_lists = _parse_generic_response(response_section)

        if isinstance(response_lists, list):
            standardized_response = [
                [standardize_formula(item) for item in sublist]
                for sublist in response_lists
                if isinstance(sublist, list)
            ]
            parsed_entry["response"] = standardized_response

    return parsed_entry


def get_model_type(filename):
    """Determine model type from filename."""
    if "openai" in filename:
        return "openai"
    if "qwen" in filename:
        return "qwen"
    if "gemini" in filename:
        return "gemini"
    if "deepseek" in filename:
        return "deepseek"
    if "llama" in filename or ("meta" in filename and "llama" in filename):
        return "llama"
    return "unknown"


def parse_results_file(file_path):
    """Parse a results file, extract data, and deduplicate entries."""
    model_type = get_model_type(os.path.basename(file_path).lower())
    response_parser = get_response_parser(model_type)
    unique_entries = {}

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        return None
    except Exception as e:
        logging.error("Error reading file %s: %s", file_path, e)
        return None

    entries = re.split(r"-----\n", content)
    for entry_content in entries:
        if not entry_content.strip():
            continue

        parsed_entry = parse_entry(entry_content.strip(), response_parser)
        if parsed_entry:
            _update_unique_entries(unique_entries, parsed_entry)

    return list(unique_entries.values())


def _update_unique_entries(unique_entries, current_entry):
    """Update the dictionary of unique entries, prioritizing more complete records."""
    target_key = current_entry["target"]
    has_current_gt = bool(current_entry["ground_truth"])
    has_current_resp = bool(current_entry["response"])
    current_score = (2 if has_current_gt else 0) + (1 if has_current_resp else 0)

    if target_key not in unique_entries:
        unique_entries[target_key] = current_entry
    else:
        existing_entry = unique_entries[target_key]
        has_existing_gt = bool(existing_entry["ground_truth"])
        has_existing_resp = bool(existing_entry["response"])
        existing_score = (2 if has_existing_gt else 0) + (1 if has_existing_resp else 0)

        if current_score > existing_score:
            unique_entries[target_key] = current_entry


def save_to_json(data, output_path):
    """Save the parsed data to a JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
        logging.info("Successfully saved JSON data to: %s", output_path)
    except (IOError, TypeError) as e:
        logging.error("Error writing JSON to file %s: %s", output_path, e)


def main():
    """Main function to process result files from a directory."""
    parser = argparse.ArgumentParser(
        description="Convert text results files to JSON format."
    )
    parser.add_argument(
        "-d", "--directory", default="results", help="Input directory for .txt files."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="results_jsons",
        help="Output directory for .json files.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        logging.error("Input directory not found: %s", args.directory)
        return

    os.makedirs(args.output, exist_ok=True)

    txt_files = [
        f
        for f in os.listdir(args.directory)
        if f.startswith("results_") and f.endswith(".txt")
    ]
    total_files = len(txt_files)
    logging.info(
        "Found %d '.txt' files to process in '%s'.", total_files, args.directory
    )

    for i, filename in enumerate(txt_files):
        logging.info("Processing file %d of %d: %s", i + 1, total_files, filename)
        input_path = os.path.join(args.directory, filename)
        parsed_data = parse_results_file(input_path)

        if parsed_data:
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(args.output, f"{base_name}.json")
            save_to_json(parsed_data, output_path)

    logging.info("Finished processing all files.")


if __name__ == "__main__":
    main()
