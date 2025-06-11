"""
Utility functions for inorganic materials synthesis planning.

This module provides utility functions for interacting with language models,
processing synthesis data, formatting prompts, and handling API responses
for both retrosynthesis and conditions prediction tasks.

Key Features:
- Language model client management and API calls
- Prompt generation for synthesis planning tasks
- Data parsing and validation utilities
- Result formatting and file I/O operations
"""

import ast
import json
import logging
import os
import re
import time
from datetime import datetime
import sys

import numpy as np
import pandas as pd
from google.genai import types


def sanitize_filename(filename):
    """Sanitize filename by replacing invalid characters with underscore"""
    # Replace characters that are invalid in Windows filenames
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, "_", filename)


def setup_logging(log_filename=None, model_name=None, start_index=0, end_index=None):
    """Set up logging configuration, including batch indices in the filename if applicable."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Sanitize model name for filename if provided
    if model_name:
        model_name = sanitize_filename(model_name)

    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name:
            log_filename = f"logs/retrosynthesis_{model_name}_{timestamp}.log"
        else:
            log_filename = f"logs/retrosynthesis_test_{timestamp}.log"
    else:
        # Ensure log file is in logs directory
        if not log_filename.startswith("logs/"):
            log_filename = os.path.join("logs", log_filename)

    # Determine batch suffix based on indices
    batch_suffix = ""
    if start_index > 0 or end_index is not None:
        batch_suffix += f"_start{start_index}"
        if end_index is not None:
            batch_suffix += f"_end{end_index}"
        else:
            batch_suffix += "_endFull"  # Indicate slicing to the end

    # Insert batch suffix before the timestamp or .log extension
    if timestamp:  # If timestamp was generated
        base_name = log_filename.replace(f"_{timestamp}.log", "")
        log_filename = f"{base_name}{batch_suffix}_{timestamp}.log"
    else:  # If a specific log_filename was provided
        base_name, ext = os.path.splitext(log_filename)
        log_filename = f"{base_name}{batch_suffix}{ext}"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(
        f"Logging initialized. Start Index: {start_index}, End Index: {end_index}"
    )  # Log indices
    return log_filename


def get_precursors(client, model_name, prompt, delay=2):
    """Get precursors from various model providers based on model name"""
    try:
        # Apply model-specific delays for Google models to respect rate limits
        if model_name.startswith("gemini"):
            # Rate limits for different Gemini models (in requests per minute)
            gemini_rate_limits = {
                "gemini-2.5-pro": 2,  # 2 RPM
                "gemini-2.0-flash-thinking": 10,  # 10 RPM
                "gemini-2.0-flash-001": 15,  # 15 RPM
                "gemini-2.0-flash-lite": 30,  # 30 RPM
                "gemini-2.0-flash-exp": 10,  # 10 RPM
                "gemini-1.5-flash": 15,  # 15 RPM
                "gemini-1.5-flash-8b": 15,  # 15 RPM
                "gemini-1.5-pro": 2,  # 2 RPM
                "gemini-embedding": 5,  # 5 RPM
            }

            # Determine which model family this is
            model_delay = delay  # Default delay if no match
            for model_family, rpm in gemini_rate_limits.items():
                if model_name.startswith(model_family):
                    # Convert RPM to seconds per request
                    model_delay = 60.0 / rpm
                    logging.info(
                        f"Using rate limit for {model_family}: {rpm} RPM ({model_delay:.2f} seconds delay)"
                    )
                    break

            logging.info(
                f"Applying delay of {model_delay:.2f} seconds for {model_name}"
            )
            time.sleep(model_delay)

        # Google (Gemini) models
        if model_name.startswith("gemini"):
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,  # The prompt already includes the system message
                config=types.GenerateContentConfig(
                    temperature=0.1,
                ),
            )
            return response.text.strip()

        # Use OpenAI-compatible API for all other models (OpenAI, xAI, OpenRouter, Mistral, Anthropic, LLAMA)
        else:
            # Models that use OpenAI-compatible chat completions API
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,  # The prompt already includes the system message
                    }
                ],
                model=model_name,
                temperature=0.1,
                stream=False,
            )
            return chat_completion.choices[0].message.content.strip()

    except (types.GoogleAPICallError, OSError) as e:
        logging.error(f"Error with API for model {model_name}: {str(e)}")
        print(f"Error with API for model {model_name}: {str(e)}")
        return None


def save_responses_to_file(
    filename,
    entry_key,
    ground_truth,
    predictions,
    model_name=None,
    dataset_type="test",
    num_few_shot=2,
    task="retrosynthesis",
):
    """
    Saves the ground truth, predictions, and other details to a file.
    For 'conditions' task, if predictions is a dict, it's saved as JSON.
    """
    try:
        # Ensure the directory exists right before opening the file
        # The filename passed should already contain the custom output directory
        results_dir = os.path.dirname(filename)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        else:
            # If filename has no directory part, default to current dir, but this shouldn't happen with the new setup
            logging.warning(
                f"Filename '{filename}' has no directory part. Saving to current directory."
            )
            # Consider if a default like '.' or 'results' is better if this case is possible

        with open(filename, "a", encoding="utf-8") as file:
            if entry_key is not None:
                file.write("-----\n")

                # Write identifier based on task
                if task == "conditions":
                    file.write(f"Reaction: {entry_key}\n")
                    # Format ground truth dictionary nicely
                    try:
                        gt_str = json.dumps(ground_truth, indent=2)
                        file.write(f"Ground Truth Conditions:\n{gt_str}\n")
                    except Exception as e:
                        logging.warning(
                            f"Could not format ground truth for {entry_key}: {e}"
                        )
                        file.write(
                            f"Ground Truth Conditions: {str(ground_truth)}\n"
                        )  # Fallback

                    # Handle predictions (should be a dict or None for conditions)
                    if predictions is None:
                        file.write(
                            "Predicted Conditions: None\n"
                        )  # Use correct key for None
                    elif isinstance(predictions, dict):
                        try:
                            formatted_response = json.dumps(predictions, indent=2)
                            file.write(f"Predicted Conditions:\n{formatted_response}\n")
                        except Exception as e:
                            logging.error(
                                f"Could not format conditions prediction as JSON for {entry_key}: {e}"
                            )
                            file.write(
                                f"Predicted Conditions: {str(predictions)}\n"
                            )  # Fallback
                    else:
                        # If prediction is not a dict (unexpected), save its string representation
                        logging.warning(
                            f"Unexpected prediction format for conditions task (expected dict, got {type(predictions)}). Saving as string."
                        )
                        file.write(f"Predicted Conditions (Raw): {str(predictions)}\n")

                else:  # retrosynthesis
                    file.write(f"Target material: {entry_key}\n")
                    file.write(f"Ground truth precursors: {ground_truth}\n")
                    # Handle predictions (list or None for retrosynthesis)
                    if predictions is None:
                        file.write("Predicted Precursors: None\n")
                    else:
                        file.write(f"Predicted Precursors: {predictions}\n")

                file.write("-----\n\n")

        logging.info(f"Response for {entry_key} saved to {filename}")

        return filename
    except (IOError, OSError) as e:
        logging.error("Error saving responses to file %s: %s", filename, e)
        return None


def read_existing_results(results_path, task):
    """Helper to read and parse existing results from a file."""
    entry_status = {}
    processed_keys_from_file = set()
    if not os.path.exists(results_path):
        return entry_status, processed_keys_from_file

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            content = f.read()
        sections = re.split(r"\\n-{5,}\\n", content.strip())

        for section_content in sections:
            if not section_content or "ACCURACY_SUMMARY" in section_content:
                continue
            # Simplified parsing logic for brevity
            if task == "retrosynthesis" and "Target material:" in section_content:
                key_match = re.search(r"Target material:\\s*(\\S+)", section_content)
                if key_match:
                    processed_keys_from_file.add(key_match.group(1))

    except (IOError, OSError) as e:
        logging.error(
            "Error reading or parsing results file %s: %s",
            results_path,
            e,
            exc_info=True,
        )

    return entry_status, processed_keys_from_file


def create_canonical_gt_key(task, ground_truth_to_save, target_formula):
    """Creates a canonical key for a given entry for tracking."""
    try:
        if task == "conditions":
            gt_dict_standardized = {
                k: v.item() if isinstance(v, np.generic) else v
                for k, v in ground_truth_to_save.items()
            }
            return json.dumps(gt_dict_standardized, sort_keys=True)
        if task == "retrosynthesis":
            return target_formula
    except (TypeError, ValueError) as e:
        logging.error("Could not create canonical GT key: %s", e, exc_info=True)
    return None


def should_skip_entry(
    canonical_gt_key, dataset_type, entry_status, processed_keys_from_file
):
    """Determines if an entry should be skipped."""
    if dataset_type == "inference" and canonical_gt_key in processed_keys_from_file:
        return True
    if dataset_type != "inference" and entry_status.get(canonical_gt_key) == "valid":
        return True
    return False


def remove_duplicates(predictions):
    """Remove duplicates while preserving the order"""
    try:
        seen = set()
        unique_predictions = []
        for pred in predictions:
            pred_set = frozenset(pred)
            if pred_set not in seen:
                unique_predictions.append(pred)
                seen.add(pred_set)
        return unique_predictions
    except TypeError as e:
        logging.error("Error removing duplicates: %s", e)
        return predictions


def extract_elements(formula):
    """Extract chemical elements from a formula"""
    try:
        element_pattern = re.compile(r"([A-Z][a-z]?)")
        elements = re.findall(element_pattern, formula)
        return set(elements)  # Return as a set to avoid duplicates
    except TypeError as e:
        logging.error("Error extracting elements from formula: %s", e)
        return set()


def validate_precursors(predictions, target_formula):
    """Validate if the precursors can form the target material based on element conservation."""
    if not predictions:  # Handle case where predictions might be None or empty
        return predictions, [], 0

    target_elements = extract_elements(target_formula)
    valid_candidates = []
    invalid_candidates = []
    elemental_check_passed_count = 0

    for candidate in predictions:
        candidate_set = set()
        for item in candidate:
            candidate_set.update(extract_elements(item))

        if target_elements.issubset(candidate_set):
            valid_candidates.append(candidate)
            elemental_check_passed_count += 1
        else:
            invalid_candidates.append(candidate)
            reason = f"Missing elements: {target_elements - candidate_set}"
            logging.debug(
                f"Validation result for {candidate}: {False}, Reason: {reason}"
            )

    return valid_candidates, invalid_candidates, elemental_check_passed_count


def load_dataset(test_file_path, max_samples=None):
    """Load dataset from CSV file."""
    try:
        # Load the dataset
        df = pd.read_csv(test_file_path)
        logging.info(f"Loaded dataset with {df.shape[0]} samples")

        # Limit samples if needed
        if max_samples is not None and max_samples > 0 and len(df) > max_samples:
            df = df.head(max_samples)
            logging.info("Limiting to %d samples for testing", max_samples)
            print(f"Limiting to {max_samples} samples for testing")

        # Convert string representation of lists to actual lists, if the column exists
        if "precursor_formulas" in df.columns:
            df["precursor_formulas"] = df["precursor_formulas"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        return df
    except (FileNotFoundError, ValueError) as e:
        logging.error("Error loading dataset: %s", e)
        print(f"Error loading dataset: {e}")
        return None


def create_retrosynthesis_prompt(
    target_formula,
    num_precursors,
    num_combinations=20,
    example_data=None,
    num_examples=None,
):
    """
    Creates a scientifically informed prompt for retrosynthesis prediction,
    potentially using few-shot examples and clearer constraints.
    """
    prompt = (
        f"You are a computational chemistry expert specializing in solid-state synthesis and retrosynthesis.\n"
        f"Your task is to identify potential precursor combinations for solid-state synthesizing the target material: '{target_formula}'.\n\n"
        f"**Requirements**:\n"
        f"1.  Propose exactly {num_precursors} precursors for each synthesis route.\n"
        f"2.  Generate {num_combinations} distinct combinations of precursor materials.\n"
        f"3.  Use standard chemical formulas ONLY (e.g., 'TiO2', 'Na2CO3').\n"
        f"4.  **Constraint Check:** Ensure each precursor combination contains ALL elements present in the target material '{target_formula}'. Assume Oxygen is available.\n"
        f"5.  **Plausibility Filter:** Prefer chemically plausible routes using reasonably common and stable laboratory reagents.\n"
        f"6.  Order the {num_combinations} combinations from the MOST plausible/common synthesis routes to the LEAST plausible/common.\n"
    )

    if example_data:
        prompt += "\n**Examples of Target -> Precursors:**\n"
        examples = example_data
        if num_examples is not None:
            examples = example_data[:num_examples]
        for example in examples:
            try:
                precursors_list = (
                    ast.literal_eval(example["precursor_formulas"])
                    if isinstance(example["precursor_formulas"], str)
                    else example["precursor_formulas"]
                )
                # Ensure the example matches the requested number of precursors if possible, or skip/note otherwise.
                if len(precursors_list) == num_precursors:
                    prompt += f"- Target: '{example['target_formula']}', Precursors: {precursors_list}\n"
            except (ValueError, SyntaxError):
                prompt += f"- Target: '{example['target_formula']}', Precursors: [Error Parsing Example]\n"

    # Create a dynamic example format based on the number of precursors
    example_format = "["
    for i in range(1, num_combinations + 1):
        if i > 1:
            example_format += ", "
        precursor_example = "["
        for j in range(1, num_precursors + 1):
            if j > 1:
                precursor_example += ", "
            precursor_example += f"'precursor{i}{chr(96+j)}'"
        precursor_example += "]"

        # Only show first 2 and last item if more than 3 combinations
        if num_combinations > 3 and i > 2 and i < num_combinations:
            if not example_format.endswith("..."):
                example_format += "..."
            i = num_combinations - 1
        else:
            example_format += precursor_example
    example_format += "]"

    prompt += (
        f"\nGenerate the list of {num_combinations} precursor combinations for the target '{target_formula}'.\n"
        f"**Output Format:** Respond ONLY with a single Python-formatted list of lists. Each inner list must contain exactly {num_precursors} precursor strings.\n"
        f"Example Output Format: {example_format}"
    )
    return prompt


def create_retrosynthesis_prompt_inference(
    target_formula, num_combinations=20, example_data=None, num_examples=None
):
    """
    Creates a scientifically informed prompt for retrosynthesis prediction,
    potentially using few-shot examples and clearer constraints.
    """
    prompt = (
        f"You are a computational chemistry expert specializing in solid-state synthesis and retrosynthesis.\n"
        f"Your task is to identify potential precursor combinations for solid-state synthesizing the target material: '{target_formula}'.\n\n"
        f"**Requirements**:\n"
        f"1.  Generate {num_combinations} distinct combinations of precursor materials.\n"
        f"2.  Use standard chemical formulas ONLY (e.g., 'TiO2', 'Na2CO3').\n"
        f"3.  **Constraint Check:** Ensure each precursor combination contains ALL elements present in the target material '{target_formula}'. Assume Oxygen and other common laboratory elements (e.g., C for carbonate sources) are available.\n"
        f"4.  **Plausibility Filter:** Prefer chemically plausible routes using reasonably common and stable laboratory reagents. A plausible route is one that uses precursors commonly found in solid-state synthesis and avoids highly unstable or rare compounds.\n"
        f"5.  Order the {num_combinations} combinations from the MOST plausible/common synthesis routes to the LEAST plausible/common.\n"
        f"6.  **Common Precursor Types:** Consider oxides (e.g., TiO2, Fe2O3), carbonates (e.g., Na2CO3, CaCO3), nitrates (e.g., KNO3, Ca(NO3)2), hydroxides (e.g., Al(OH)3), and other standard laboratory reagents.\n"
        f"7.  **No Gases:** Do not include 'O2' in the precursor combinations.\n"
        f"7.  If the target material is not suitable for solid-state synthesis, respond with False as a boolean.\n"
    )

    # Add the MoF₅ example as a False example
    prompt += "\n**Examples of Target -> Precursors:**\n"
    prompt += "- Target: 'MoF₅', Precursors: [False] #only synthesizable via gas-solid reaction, not suitable for conventional solid-state synthesis\n"

    if example_data:
        # Use only the first num_examples if specified
        examples = example_data
        if num_examples is not None:
            examples = example_data[:num_examples]
        for example in examples:
            try:
                precursors_list = (
                    ast.literal_eval(example["precursor_formulas"])
                    if isinstance(example["precursor_formulas"], str)
                    else example["precursor_formulas"]
                )
                prompt += f"- Target: '{example['target_formula']}', Precursors: {precursors_list}\n"
            except (ValueError, SyntaxError):
                prompt += f"- Target: '{example['target_formula']}', Precursors: [Error Parsing Example]\n"
        prompt += "\nNote: Ensure the quality and consistency of the example data to prevent parsing issues.\n"

    # Create a dynamic example format based on the number of combinations
    example_format = "["
    for i in range(1, min(4, num_combinations + 1)):
        if i > 1:
            example_format += ", "
        precursor_example = "["
        # Use variable number of precursors in the example
        for j in range(1, 4):  # Show up to 3 precursors as example
            if j > 1:
                precursor_example += ", "
            precursor_example += f"'precursor{i}{chr(96+j)}'"
        precursor_example += "]"

        # Only show first 2 and last item if more than 3 combinations
        if num_combinations > 3 and i == 3:
            example_format += ", ..., "
            precursor_example = "["
            for j in range(1, 4):
                if j > 1:
                    precursor_example += ", "
                precursor_example += f"'precursor{num_combinations}{chr(96+j)}'"
            precursor_example += "]"
            example_format += precursor_example
            break
        else:
            example_format += precursor_example
    example_format += "]"

    prompt += (
        f"\nGenerate the list of {num_combinations} precursor combinations for the target '{target_formula}'.\n"
        f"**Output Format:** Respond ONLY with a single Python-formatted list of lists. Each inner list should contain the precursor strings. Typically, 2-4 precursors per combination are expected.\n"
        f"Example Output Format: {example_format}\n\n"
        f"**Important:** Ensure all combinations are chemically valid and contain all elements needed to synthesize the target material. Do not include any explanations or text outside the Python list format."
    )
    return prompt


def create_retrosynthesis_conditions_prompt(
    reaction_equation, example_data=None, num_examples=3, few_shot_examples=None
):
    """
    Creates a scientifically informed prompt for predicting synthesis conditions,
    using the full reaction equation as input.

    Uses examples from few_shot_reactions.csv by default.
    """
    prompt = (
        f"You are a computational chemistry expert specializing in solid-state synthesis.\n"
        f"Assume there is only one sintering and one calcination step involved.\n"
        f"Your task is to predict the optimal synthesis conditions for the following chemical reaction:\n"
        f"{reaction_equation}\n\n"
        f"**Required Conditions to Predict**:\n"
        f"1. Sintering Temperature (in °C)\n"
        f"2. Sintering Time (in hours)\n"
        f"3. Calcination Temperature (in °C)\n"
        f"4. Calcination Time (in hours)\n\n"
        f"**Guidelines for Prediction**:\n"
        f"- Base your predictions on established solid-state chemistry principles.\n"
        f"- Provide scientifically plausible values within typical laboratory ranges.\n"
        f"- Assume there is only one sintering and one calcination step involved.\n"
    )

    # Helper function to format an example for the prompt
    def format_example(example):
        # Use the Reaction field directly
        reaction = example.get("Reaction", "")

        # Extract synthesis conditions
        heating_temps = example.get("Heating Temperatures", "[[]]")
        heating_times = example.get("Heating Times", "[[]]")
        sintering_temp = example.get("Sintering Temperature", "")
        sintering_time = example.get("Sintering Time", "")
        calcination_temp = example.get("Calcination Temperature", "")
        calcination_time = example.get("Calcination Time", "")

        # Use N/A for missing numeric values
        sintering_temp_str = (
            sintering_temp
            if pd.notna(sintering_temp) and str(sintering_temp).strip()
            else "N/A"
        )
        sintering_time_str = (
            sintering_time
            if pd.notna(sintering_time) and str(sintering_time).strip()
            else "N/A"
        )
        calcination_temp_str = (
            calcination_temp
            if pd.notna(calcination_temp) and str(calcination_temp).strip()
            else "N/A"
        )
        calcination_time_str = (
            calcination_time
            if pd.notna(calcination_time) and str(calcination_time).strip()
            else "N/A"
        )

        return (
            f"- Reaction: {reaction}\n"
            f"  Sintering Temperature (°C): {sintering_temp_str}\n"
            f"  Sintering Time (hours): {sintering_time_str}\n"
            f"  Calcination Temperature (°C): {calcination_temp_str}\n"
            f"  Calcination Time (hours): {calcination_time_str}\n\n"
        )

    # Add examples to the prompt
    prompt += "\n**Examples of Synthesis Conditions:**\n"

    # First try to use examples from few_shot_reactions.csv
    # few_shot_path = 'datasets/llm_datasets/few_shot_dataset/few_shot_reactions.csv'
    examples_added = 0

    few_shot_df = example_data

    # Shuffle the dataframe to get random examples
    shuffled_df = few_shot_df.sample(frac=1).reset_index(drop=True)

    examples_added_from_csv = 0
    for _, example in shuffled_df.iterrows():
        if examples_added_from_csv >= num_examples:
            break  # Stop if we have enough examples

        try:
            prompt += format_example(example)
            examples_added_from_csv += 1  # Increment only on success
        except (ValueError, SyntaxError, KeyError) as e:
            # Skip examples that can't be parsed
            logging.warning(
                f"Skipping few-shot CSV example due to formatting error: {e}"
            )
            continue
    examples_added = examples_added_from_csv  # Update the main counter

    # Fall back to provided example_data if needed
    if examples_added < num_examples and example_data:
        needed = num_examples - examples_added
        for example in example_data[:needed]:  # Take only the remaining needed examples
            try:
                # Assuming example_data also has a 'Reaction' field now
                prompt += format_example(example)
                examples_added += 1  # Increment the main counter here too
            except (ValueError, SyntaxError, KeyError) as e:
                logging.warning(f"Skipping provided example data due to error: {e}")
                continue

    # Add output instructions
    prompt += (
        f"\nPredict the synthesis conditions for the reaction:\n{reaction_equation}\n\n"
        f"**Output Format:** Respond *exactly* with a Python dict matching this template—"
        f"replace each `<…>` with your predicted number (float):\n\n"
        f"```python\n"
        f"{{\n"
        f"  'Sintering Temperature': <SINTER_TEMP>,   # °C (float)\n"
        f"  'Sintering Time': <SINTER_TIME>,         # hours (float)\n"
        f"  'Calcination Temperature': <CALC_TEMP>,   # °C (float)\n"
        f"  'Calcination Time': <CALC_TIME>          # hours (float)\n"
        f"}}\n"
        f"```"
    )

    return prompt


def process_prediction_response(predicted_text):
    """Extract and clean the prediction list from model response"""
    try:
        # Clean up response - remove any text before and after the actual list
        match = re.search(r"\[.*\]", predicted_text, re.DOTALL)
        if match:
            cleaned_text = match.group(0)
        else:
            logging.warning(f"Could not extract list from response")
            return None

        # Remove ellipses if present
        if "..." in cleaned_text:
            cleaned_text = cleaned_text.replace("...", "")

        # Convert to Python list
        precursor_combinations = ast.literal_eval(cleaned_text)
        return precursor_combinations

    except (ValueError, SyntaxError) as e:
        logging.error(f"Error processing model response: {e}")
        return None


def process_conditions_response(predicted_text):
    """Extract and clean the synthesis conditions dictionary from model response"""
    try:
        # Clean up response - remove any text before and after the actual dictionary
        match = re.search(r"\{.*\}", predicted_text, re.DOTALL)
        if match:
            cleaned_text = match.group(0)
        else:
            logging.warning(f"Could not extract dictionary from conditions response")
            return None

        # Remove comments if present
        cleaned_text = re.sub(r"#.*$", "", cleaned_text, flags=re.MULTILINE)

        # Convert to Python dictionary
        try:
            conditions_dict = ast.literal_eval(cleaned_text)
        except (ValueError, SyntaxError):
            # Try to fix common issues with the dictionary syntax
            # Replace single quotes with double quotes for JSON compatibility
            fixed_text = cleaned_text.replace("'", '"')
            # Try to parse as JSON
            # json already imported at module level

            try:
                conditions_dict = json.loads(fixed_text)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse conditions as Python dict or JSON: {e}")
                return None

        # Validate the required keys
        required_keys = [
            "Sintering Temperature",
            "Sintering Time",
            "Calcination Temperature",
            "Calcination Time",
        ]

        # Add missing keys with null values
        for key in required_keys:
            if key not in conditions_dict:
                conditions_dict[key] = None

        return conditions_dict

    except (ValueError, SyntaxError) as e:
        logging.error(f"Error processing conditions response: {e}")
        return None


def extract_predicted_values(raw_prediction):
    """Extract predicted values from the raw prediction text."""
    # For python dictionary format (with triple backticks)
    if "```python" in raw_prediction:
        # Extract content between python code blocks
        match = re.search(
            r"```python\s*(\{[\s\S]*?\})\s*```", raw_prediction, re.DOTALL
        )
        if match:
            try:
                # Extract the dictionary string
                dict_str = match.group(1)
                # Remove comments line by line
                lines = dict_str.split("\n")
                cleaned_lines = [re.sub(r"#.*", "", line) for line in lines]
                cleaned_dict_str = "\n".join(cleaned_lines)

                # Clean up the text (replace single quotes with double quotes)
                json_str = cleaned_dict_str.replace("'", '"')
                # Handle potential trailing commas which are valid in Python but not in JSON
                json_str = re.sub(r",\s*}", "}", json_str)
                json_str = re.sub(
                    r",\s*]", "]", json_str
                )  # Handle trailing comma in lists if any
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.warning(
                    f"JSON decode error in ```python block: {e}"
                )  # Changed print to logging
                # logging.warning(f"Raw text after cleaning: {cleaned_dict_str}") # Optional: log cleaned text
                return {}

    # Try to find any dictionary-like structure if ```python block fails or is not present
    pattern = r"\{[\s\S]*?\}"
    match = re.search(pattern, raw_prediction)
    if match:
        try:
            # Extract the dictionary string
            dict_str = match.group()
            # Remove comments line by line
            lines = dict_str.split("\n")
            cleaned_lines = [re.sub(r"#.*", "", line) for line in lines]
            cleaned_dict_str = "\n".join(cleaned_lines)

            # Clean up the text
            json_str = cleaned_dict_str.replace("'", '"')
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(
                r",\s*]", "]", json_str
            )  # Handle trailing comma in lists if any
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.warning(
                f"Failed to parse dictionary-like structure: {e}"
            )  # Changed print to logging
            # logging.warning(f"Problematic structure: {match.group()}") # Optional: log structure
            return {}

    logging.warning(f"Could not extract any dictionary from raw prediction.")
    return {}


def validate_conditions_dict(prediction_dict):
    """
    Checks if a parsed prediction dictionary for the 'conditions' task contains
    valid numeric values for the required parameters.
    """
    required_params = [
        "Sintering Temperature",
        "Sintering Time",
        "Calcination Temperature",
        "Calcination Time",
    ]

    # Validate the dictionary directly, don't re-parse
    if not prediction_dict or not isinstance(prediction_dict, dict):
        logging.debug(f"Validation failed: Input is not a valid dictionary.")
        return False

    for param in required_params:
        if param not in prediction_dict:
            logging.debug(f"Validation failed: Required parameter '{param}' missing.")
            return False

        value = prediction_dict.get(
            param
        )  # Use .get for safety, though check above should suffice
        if value is None:
            logging.debug(f"Validation failed: Parameter '{param}' is None.")
            return False

        try:
            num_value = float(value)
            if np.isnan(num_value):
                logging.debug(f"Validation failed: Parameter '{param}' is NaN.")
                return False
        except (ValueError, TypeError):
            logging.debug(
                f"Validation failed: Parameter '{param}' ('{value}') is not a valid number."
            )
            return False

    # If all checks pass
    logging.debug(f"Validation successful for prediction dictionary.")
    return True


def load_examples(file_path, num_examples):
    """Load examples from a dataset file."""
    try:
        if not os.path.exists(file_path):
            logging.warning("File %s not found.", file_path)
            return None

        df = pd.read_csv(file_path)
        if len(df) < num_examples:
            logging.warning(
                "Not enough examples in %s. Using %d examples.", file_path, len(df)
            )
            num_examples = len(df)

        examples = df.head(num_examples).to_dict("records")
        logging.info("Loaded %d examples from %s", len(examples), file_path)
        return examples
    except (FileNotFoundError, ValueError) as e:
        logging.error("Error loading examples from %s: %s", file_path, e)
        return None
