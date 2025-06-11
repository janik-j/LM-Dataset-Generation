"""
Predictions module for inorganic materials synthesis planning.

This module provides functionality for running predictions using various language models
for retrosynthesis and conditions prediction tasks. It supports multiple AI providers
including Google, OpenAI, xAI, and OpenRouter.

Key Features:
- Support for multiple language models from different providers
- Retrosynthesis prediction with few-shot learning
- Conditions prediction for synthesis planning
- Comprehensive evaluation and result saving
"""

import argparse
import ast
import json
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from openai import OpenAI

import utils

# --- Setup ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- API Clients ---
google_client = (
    genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    if os.getenv("GEMINI_API_KEY")
    else None
)
openai_client = (
    OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
)
xai_client = (
    OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
    if os.getenv("XAI_API_KEY")
    else None
)
openrouter_client = (
    OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    if os.getenv("OPENROUTER_API_KEY")
    else None
)

# --- Constants ---
MAX_SAMPLES = None
AVAILABLE_MODELS = {
    # Google models
    "gemini-2.5-pro-exp-03-25": {"client": google_client},
    "gemini-2.0-flash-thinking-exp-01-21": {"client": google_client},
    "gemini-2.0-flash-001": {"client": google_client},
    # OpenAI models
    "gpt-4.5-preview-2025-02-27": {"client": openai_client},
    "o3-mini-high": {"client": openai_client},
    "o1-2024-12-17": {"client": openai_client},
    "chatgpt-4o-lates-20250179": {"client": openai_client},
    "o3-mini": {"client": openai_client},
    "gpt-4o": {"client": openai_client},
    "gpt-4.1": {"client": openai_client},
    # OpenRouter models
    "deepseek/deepseek-chat-v3-0324": {"client": openrouter_client},
    "deepseek/deepseek-r1": {"client": openrouter_client},
    "qwen/qwen2.5-vl-72b-instruct": {"client": openrouter_client},
    "meta-llama/llama-3.3-70b-instruct": {"client": openrouter_client},
    "meta-llama/llama-4-maverick": {"client": openrouter_client},
    "meta-llama/llama-4-scout": {"client": openrouter_client},
    "openai/o3-mini": {"client": openrouter_client},
    "mistralai/mistral-small-3.1-24b-instruct": {"client": openrouter_client},
    "openrouter/quasar-alpha": {"client": openrouter_client},
    "openrouter/optimus-alpha": {"client": openrouter_client},
    "google/gemini-2.5-pro-preview-03-25": {"client": openrouter_client},
    "google/gemini-2.0-flash-001": {"client": openrouter_client},
    "meta-llama/llama-4-maverick:free": {"client": openrouter_client},
    "openai/gpt-4.1": {"client": openrouter_client},
    "x-ai/grok-3-mini-beta": {"client": openrouter_client},
}

RETRO_DATA_PATH = "datasets/dataset_w_candidates_w_val_unique_systems_hard"
RETRO_TEST_FILE = "test_reduced.csv"
RETRO_VAL_FILE = "val.csv"
RETRO_TRAIN_FILE = "train.csv"
INFERENCE_DATA_PATH = "datasets/materials_project_inference/mp_synthesized.csv"
COND_DATA_PATH = "datasets/condition_llm_dataset"
COND_TEST_FILE = "test_data.csv"
COND_VAL_FILE = "val_data.csv"
COND_FEW_SHOT_FILE = "train_data.csv"
COND_INFERENCE_FILE = "predictions_mat_discovery.csv"


def run_prediction_test(
    task,
    model_info,
    test_file_path,
    max_samples,
    num_few_shot,
    dataset_type="test",
    no_fixed_precursor_len=False,
    start_index=0,
    end_index=None,
    output_dir="results",
):
    """Run predictions for a given model, task, and dataset."""
    try:
        model_name = model_info["name"]
        safe_model_name = utils.sanitize_filename(model_name)
        os.makedirs(output_dir, exist_ok=True)

        few_shot_examples, few_shot_examples_conditions_df = _load_few_shot_examples(
            task, num_few_shot
        )

        df = _load_and_batch_dataset(
            test_file_path, max_samples, start_index, end_index, dataset_type, task
        )
        if df is None:
            return

        results_filename = (
            f"results_{task}_{safe_model_name}_{dataset_type}_"
            f"fewshot{num_few_shot}_fixed_prec_len_{not no_fixed_precursor_len}_"
            f"start{start_index}_end{end_index if end_index is not None else 'Full'}.txt"
        )
        results_path = os.path.join(output_dir, results_filename)

        _process_dataset(
            df,
            task,
            model_info,
            dataset_type,
            few_shot_examples,
            few_shot_examples_conditions_df,
            no_fixed_precursor_len,
            results_path,
        )

    except Exception as e:
        logging.error(
            "Error in main execution for task '%s': %s", task, e, exc_info=True
        )
        print(f"An error occurred during task '{task}': {e}")


def _load_few_shot_examples(task, num_few_shot):
    """Load few-shot examples based on the task."""
    if task == "conditions":
        few_shot_file_path = os.path.join(COND_DATA_PATH, COND_FEW_SHOT_FILE)
        try:
            df = pd.read_csv(few_shot_file_path)
            logging.info(
                "Conditions task will use examples from: %s", few_shot_file_path
            )
            num_to_save = min(num_few_shot, len(df))
            examples = df.head(num_to_save).to_dict("records")
            return examples, df
        except (FileNotFoundError, IOError, OSError):
            logging.warning(
                "Conditions few-shot file not found at %s. Proceeding without.",
                few_shot_file_path,
            )
            return [], None

    few_shot_file_path = os.path.join(RETRO_DATA_PATH, RETRO_TRAIN_FILE)
    examples = utils.load_examples(few_shot_file_path, num_few_shot)
    if examples is None:
        logging.warning(
            "Could not load retrosynthesis examples from %s.", few_shot_file_path
        )
        return [], None
    return examples, None


def _load_and_batch_dataset(
    file_path, max_samples, start_index, end_index, dataset_type, task
):
    """Load and apply batching to a dataset."""
    if not os.path.exists(file_path):
        logging.warning(
            "%s file %s not found for task '%s'.",
            dataset_type.capitalize(),
            file_path,
            task,
        )
        return None

    df = utils.load_dataset(file_path, max_samples)
    if df is None:
        return None

    if start_index > 0 or end_index is not None:
        original_size = len(df)
        start_idx = max(0, min(start_index, original_size))
        end_idx = (
            max(start_idx, min(end_index, original_size))
            if end_index is not None
            else original_size
        )
        df = df.iloc[start_idx:end_idx]
        logging.info(
            "Applied batching: start=%d, end=%s. Processing %d of %d samples.",
            start_idx,
            end_idx,
            len(df),
            original_size,
        )
        if df.empty:
            return None
    return df


def _process_dataset(
    df,
    task,
    model_info,
    dataset_type,
    few_shot_examples,
    few_shot_examples_conditions_df,
    no_fixed_precursor_len,
    results_path,
):
    """Iterate through a dataset and process each entry."""
    entry_status, p_keys_file = utils.read_existing_results(results_path, task)
    p_keys_run = set()
    processed_count, skipped_count = 0, 0

    for index, row in df.iterrows():
        processed, _ = _process_entry(
            row,
            index,
            task,
            model_info,
            dataset_type,
            few_shot_examples,
            few_shot_examples_conditions_df,
            no_fixed_precursor_len,
            entry_status,
            p_keys_file,
            p_keys_run,
            results_path,
        )
        if processed:
            processed_count += 1
        else:
            skipped_count += 1

    _log_task_completion(
        task, dataset_type, processed_count, skipped_count, results_path
    )


def _process_entry(
    row,
    index,
    task,
    model_info,
    dataset_type,
    few_shot_examples,
    few_shot_examples_conditions_df,
    no_fixed_precursor_len,
    entry_status,
    processed_keys_from_file,
    processed_gt_keys_in_this_run,
    results_path,
):
    """Process a single entry from the dataset."""
    (entry_key, gt_to_save, target_formula, gt_precursors) = _extract_data_from_row(
        row, task
    )
    if not entry_key:
        return False, None

    canonical_key = utils.create_canonical_gt_key(task, gt_to_save, target_formula)
    if not canonical_key:
        return False, None

    if utils.should_skip_entry(
        canonical_key, dataset_type, entry_status, processed_keys_from_file
    ):
        return False, None

    processed_gt_keys_in_this_run.add(canonical_key)

    prompt = _create_prompt(
        task,
        target_formula,
        gt_precursors,
        no_fixed_precursor_len,
        entry_key,
        few_shot_examples,
        few_shot_examples_conditions_df,
    )

    predicted_text = utils.get_precursors(
        model_info["client"], model_info["name"], prompt
    )

    _process_and_save_prediction(
        predicted_text,
        task,
        results_path,
        entry_key,
        gt_to_save,
        model_info["name"],
        dataset_type,
        len(few_shot_examples),
    )

    return True, predicted_text


def _extract_data_from_row(row, task):
    """Extract and validate data from a DataFrame row based on the task."""
    if task == "conditions":
        reaction_equation = row.get("Reaction", "")
        return reaction_equation, row.to_dict(), None, None
    if task == "retrosynthesis":
        target_formula = row.get("target_formula", "")
        precursors_raw = row.get("precursor_formulas", [])
        if not target_formula or not precursors_raw:
            return None, None, None, None
        try:
            precursors = (
                ast.literal_eval(precursors_raw)
                if isinstance(precursors_raw, str)
                else precursors_raw
            )
            return target_formula, precursors, target_formula, precursors
        except (ValueError, SyntaxError):
            return None, None, None, None
    return None, None, None, None


def _create_prompt(
    task,
    target_formula,
    ground_truth_precursors,
    no_fixed_precursor_len,
    entry_key,
    few_shot_examples,
    few_shot_examples_conditions_df,
):
    """Creates the appropriate prompt for the given task."""
    num_examples = len(few_shot_examples) if few_shot_examples else 0
    if task == "conditions":
        return utils.create_retrosynthesis_conditions_prompt(
            entry_key, few_shot_examples_conditions_df, num_examples
        )
    if no_fixed_precursor_len:
        return utils.create_retrosynthesis_prompt_inference(
            target_formula, 20, few_shot_examples, num_examples
        )
    return utils.create_retrosynthesis_prompt(
        target_formula,
        len(ground_truth_precursors),
        20,
        few_shot_examples,
        num_examples,
    )


def _process_and_save_prediction(
    predicted_text,
    task,
    results_path,
    entry_key,
    ground_truth_to_save,
    model_name,
    dataset_type,
    num_few_shot,
):
    """Process the model's prediction and save it to a file."""
    processed_data = predicted_text
    if task == "conditions" and predicted_text is not None:
        try:
            processed_data = utils.extract_predicted_values(predicted_text)
        except Exception as e:
            logging.error(
                "Error calling utils.extract_predicted_values for %s: %s", entry_key, e
            )
            processed_data = None

    utils.save_responses_to_file(
        results_path,
        entry_key,
        ground_truth_to_save,
        processed_data,
        model_name,
        dataset_type,
        num_few_shot,
        task,
    )


def _log_task_completion(
    task, dataset_type, processed_count, skipped_count, results_path
):
    """Log the completion summary of a task."""
    summary = f"Task '{task}' ({dataset_type}) completed. Processed: {processed_count}, Skipped: {skipped_count}"
    logging.info(summary)
    print(f"\n{summary}")
    print(f"Results saved to {results_path}")


def run_tests(
    task,
    model_names,
    test_set,
    val_set,
    max_samples,
    num_few_shot,
    no_fixed_precursor_len,
    inference,
    start_index,
    end_index,
    output_dir,
):
    """Dispatch prediction runs based on user arguments."""
    models_to_process = _select_models_to_process(model_names)
    if not models_to_process:
        print("No valid models selected or configured.")
        return

    if inference:
        _run_inference_mode(
            task,
            models_to_process,
            max_samples,
            num_few_shot,
            no_fixed_precursor_len,
            start_index,
            end_index,
            output_dir,
        )
    else:
        _run_evaluation_mode(
            task,
            models_to_process,
            test_set,
            val_set,
            max_samples,
            num_few_shot,
            no_fixed_precursor_len,
            start_index,
            end_index,
            output_dir,
        )


def _select_models_to_process(model_names):
    """Select and validate models to be processed."""
    models_to_process = []
    target_models = model_names if model_names else AVAILABLE_MODELS.keys()
    for name in target_models:
        if name in AVAILABLE_MODELS:
            if AVAILABLE_MODELS[name]["client"]:
                models_to_process.append(
                    {"name": name, "client": AVAILABLE_MODELS[name]["client"]}
                )
            else:
                logging.warning("API Key missing for model '%s'. Skipping.", name)
        else:
            logging.warning("Model '%s' not found. Skipping.", name)
    return models_to_process


def _run_inference_mode(
    task,
    models,
    max_samples,
    num_few_shot,
    no_fixed_precursor_len,
    start_index,
    end_index,
    output_dir,
):
    """Run the prediction task in inference mode."""
    logging.info("Running INFERENCE MODE for Task: '%s'", task)
    path = (
        INFERENCE_DATA_PATH
        if task == "retrosynthesis"
        else os.path.join(COND_DATA_PATH, COND_INFERENCE_FILE)
    )
    if not os.path.exists(path):
        logging.error("Inference dataset not found at %s. Skipping.", path)
        return
    for model in models:
        run_prediction_test(
            task,
            model,
            path,
            max_samples,
            num_few_shot,
            "inference",
            no_fixed_precursor_len,
            start_index,
            end_index,
            output_dir,
        )


def _run_evaluation_mode(
    task,
    models,
    test_set,
    val_set,
    max_samples,
    num_few_shot,
    no_fixed_precursor_len,
    start_index,
    end_index,
    output_dir,
):
    """Run the prediction task in evaluation mode."""
    logging.info("Running EVALUATION MODE for Task: '%s'", task)
    path, test_f, val_f = (
        (COND_DATA_PATH, COND_TEST_FILE, COND_VAL_FILE)
        if task == "conditions"
        else (RETRO_DATA_PATH, RETRO_TEST_FILE, RETRO_VAL_FILE)
    )
    for model in models:
        if test_set:
            _run_on_dataset(
                "test",
                task,
                model,
                path,
                test_f,
                max_samples,
                num_few_shot,
                no_fixed_precursor_len,
                start_index,
                end_index,
                output_dir,
            )
        if val_set:
            _run_on_dataset(
                "validation",
                task,
                model,
                path,
                val_f,
                max_samples,
                num_few_shot,
                no_fixed_precursor_len,
                start_index,
                end_index,
                output_dir,
            )


def _run_on_dataset(
    dataset_type,
    task,
    model,
    data_path,
    file_name,
    max_samples,
    num_few_shot,
    no_fixed_precursor_len,
    start_index,
    end_index,
    output_dir,
):
    """Run prediction on a specific dataset file."""
    file_path = os.path.join(data_path, file_name)
    if os.path.exists(file_path):
        run_prediction_test(
            task,
            model,
            file_path,
            max_samples,
            num_few_shot,
            dataset_type,
            no_fixed_precursor_len,
            start_index,
            end_index,
            output_dir,
        )
    else:
        logging.warning("File not found at %s. Skipping.", file_path)


def list_available_models():
    """Lists all available models, grouped by provider."""
    print("\n=== Available Models ===")
    providers = {
        "Google": [n for n in AVAILABLE_MODELS if n.startswith("gemini")],
        "OpenAI": [n for n in AVAILABLE_MODELS if n.startswith("gpt")],
        "OpenRouter": [n for n in AVAILABLE_MODELS if "/" in n],
    }
    for provider, models in providers.items():
        if models:
            print(f"\n{provider} Models:")
            for model in models:
                status = (
                    "Available"
                    if AVAILABLE_MODELS[model]["client"]
                    else "API Key Missing"
                )
                print(f"  - {model} ({status})")
    print("\n======================\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrosynthesis and Conditions Prediction Framework"
    )
    parser.add_argument(
        "--task",
        default="retrosynthesis",
        choices=["retrosynthesis", "conditions"],
        help="Task to perform.",
    )
    parser.add_argument(
        "--model", dest="models", action="append", help="Specific model(s) to test."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List all available models."
    )
    parser.add_argument("--test", action="store_true", help="Run on test dataset.")
    parser.add_argument("--val", action="store_true", help="Run on validation dataset.")
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to test."
    )
    parser.add_argument(
        "--few-shot-examples",
        type=int,
        default=2,
        help="Number of few-shot examples.",
    )
    parser.add_argument(
        "--no-fixed-precursor-len",
        action="store_true",
        help="Use variable precursor length for retrosynthesis.",
    )
    parser.add_argument(
        "--inference", action="store_true", help="Run on the inference dataset."
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Start index in the dataset."
    )
    parser.add_argument("--end-index", type=int, help="End index in the dataset.")
    parser.add_argument(
        "--output-dir", default="results", help="Directory to save results."
    )
    args = parser.parse_args()
    if not (args.test or args.val or args.inference):
        args.val = True
    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.list_models:
        list_available_models()
        sys.exit(0)

    run_tests(
        task=args.task,
        model_names=args.models,
        test_set=args.test,
        val_set=args.val,
        max_samples=args.max_samples,
        num_few_shot=args.few_shot_examples,
        no_fixed_precursor_len=args.no_fixed_precursor_len,
        inference=args.inference,
        start_index=args.start_index,
        end_index=args.end_index,
        output_dir=args.output_dir,
    )
