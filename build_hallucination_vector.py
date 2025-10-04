# 1_build_hallucination_vector.py
"""
Step 1: Build the Hallucination Vector (v_halluc).

This script replicates the logic from the first notebook:
1.  Loads trait data defining positive (hallucinating) and negative (non-hallucinating) prompts.
2.  Generates responses for 20 questions under both positive and negative instructions.
3.  Uses an LLM judge to score the responses for hallucination and coherence. This process
    is resilient and saves progress in batches.
4.  Filters the results to find "effective pairs" where the model's behavior perfectly
    aligned with the instructions.
5.  Extracts the hidden-state activations from the target layer for these effective pairs.
6.  Computes the mean activation for positive and negative responses and calculates their
    difference to create the final "hallucination vector".
7.  Saves the vector as `v_halluc.pt`.
"""
import os
import json
import random
import pandas as pd
import torch
from tqdm import tqdm

import config
import utils

def load_and_parse_trait_data(file_path):
    """Loads a JSON file containing persona trait data and parses it."""
    print(f"Loading trait data from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        instructions = data.get("instruction", [])
        pos_prompts = [item['pos'] for item in instructions if 'pos' in item]
        neg_prompts = [item['neg'] for item in instructions if 'neg' in item]
        questions = data.get("questions", [])
        
        print(f"Successfully loaded {len(pos_prompts)} positive prompts, {len(neg_prompts)} negative prompts, and {len(questions)} questions.")
        return pos_prompts, neg_prompts, questions
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {file_path}")
        return [], [], []
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {file_path}")
        return [], [], []

def generate_answer(system_prompt, user_question, model, tokenizer):
    """Generates an answer from the model given a system and user prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500, use_cache=True)
    
    response = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response

def extract_layer_16_activations(system_prompt, user_question, answer, model, tokenizer):
    """Extracts the mean activation of the first 5 response tokens from Layer 16."""
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_len = tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]

    full_messages = prompt_messages + [{"role": "assistant", "content": answer}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_16_hidden_states = outputs.hidden_states[config.TARGET_LAYER]
    response_activations = layer_16_hidden_states[:, prompt_len:, :]
    
    num_tokens_to_average = min(5, response_activations.shape[1])
    if num_tokens_to_average == 0:
        return torch.zeros(model.config.hidden_size, dtype=torch.float16).cpu()

    mean_activations = response_activations[:, :num_tokens_to_average, :].mean(dim=1)
    return mean_activations.squeeze().detach().cpu()

def main():
    """Main execution function for building the hallucination vector."""
    print("--- Step 1: Building the Hallucination Vector ---")
    
    # 1. Setup
    secrets = utils.load_secrets()
    api_key = secrets.get('SCALEDOWN_API_KEY')
    if not api_key:
        print("ERROR: SCALEDOWN_API_KEY not found. Cannot proceed with judging. Exiting.")
        return

    model, tokenizer = utils.load_model_and_tokenizer()
    pos_prompts, neg_prompts, questions = load_and_parse_trait_data(config.HALLUCINATING_TRAIT_DATA_PATH)

    if not questions:
        print("ERROR: No questions loaded. Exiting.")
        return

    # 2. Resilient Generation and Judging Loop
    print(f"\nStarting generation and judging loop. Results will be saved to {config.JUDGED_ANSWERS_PATH}")
    processed_questions = utils.load_processed_prompts(config.JUDGED_ANSWERS_PATH, 'question')
    print(f"Found {len(processed_questions)} already processed questions. Resuming...")

    results_data = []
    if os.path.exists(config.JUDGED_ANSWERS_PATH):
        results_data = pd.read_csv(config.JUDGED_ANSWERS_PATH).to_dict('records')

    try:
        with tqdm(total=len(questions), desc="Processing Questions") as pbar:
            pbar.update(len(processed_questions))
            for i, question in enumerate(questions):
                if question in processed_questions:
                    continue

                pos_system_prompt = random.choice(pos_prompts)
                neg_system_prompt = random.choice(neg_prompts)

                pos_answer = generate_answer(pos_system_prompt, question, model, tokenizer)
                neg_answer = generate_answer(neg_system_prompt, question, model, tokenizer)

                pos_hallucination_score = utils.get_judge_score(question, pos_answer, 'hallucination', api_key)
                pos_coherence_score = utils.get_judge_score(question, pos_answer, 'coherence', api_key)
                neg_hallucination_score = utils.get_judge_score(question, neg_answer, 'hallucination', api_key)
                neg_coherence_score = utils.get_judge_score(question, neg_answer, 'coherence', api_key)
                
                results_data.append({
                    "question": question, "pos_system_prompt": pos_system_prompt, "pos_answer": pos_answer,
                    "pos_hallucination_score": pos_hallucination_score, "pos_coherence_score": pos_coherence_score,
                    "neg_system_prompt": neg_system_prompt, "neg_answer": neg_answer,
                    "neg_hallucination_score": neg_hallucination_score, "neg_coherence_score": neg_coherence_score
                })

                if (len(results_data) % config.BATCH_SIZE_GENERATE == 0) or (i + 1 == len(questions)):
                    pd.DataFrame(results_data).to_csv(config.JUDGED_ANSWERS_PATH, index=False)
                    pbar.set_postfix({"Saved at": i+1})
                
                pbar.update(1)

    except Exception as e:
        print(f"An error occurred during generation/judging: {e}")
    finally:
        pd.DataFrame(results_data).to_csv(config.JUDGED_ANSWERS_PATH, index=False)
        print("Final results saved.")

    # 3. Filter for Effective Pairs
    print("\nFiltering for effective pairs...")
    judged_df = pd.read_csv(config.JUDGED_ANSWERS_PATH)
    mask = (
        (judged_df['pos_hallucination_score'] > config.POS_HALLUCINATION_THRESHOLD) &
        (judged_df['neg_hallucination_score'] < config.NEG_HALLUCINATION_THRESHOLD) &
        (judged_df['pos_coherence_score'] > config.COHERENCE_THRESHOLD) &
        (judged_df['neg_coherence_score'] > config.COHERENCE_THRESHOLD)
    )
    effective_df = judged_df[mask].copy().reset_index(drop=True)
    print(f"Found {len(effective_df)} effective pairs out of {len(judged_df)} total pairs.")

    if len(effective_df) == 0:
        print("ERROR: No effective pairs found. Cannot build vector. Exiting.")
        return

    # 4. Activation Extraction
    print("\nExtracting activations from effective pairs...")
    pos_dir = os.path.join(config.ARTIFACTS_DIR, "activations", "positive")
    neg_dir = os.path.join(config.ARTIFACTS_DIR, "activations", "negative")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    for index, row in tqdm(effective_df.iterrows(), total=len(effective_df), desc="Extracting Activations"):
        pos_act_path = os.path.join(pos_dir, f"activation_{index}.pt")
        neg_act_path = os.path.join(neg_dir, f"activation_{index}.pt")

        if not os.path.exists(pos_act_path):
            pos_activations = extract_layer_16_activations(row['pos_system_prompt'], row['question'], row['pos_answer'], model, tokenizer)
            torch.save(pos_activations, pos_act_path)

        if not os.path.exists(neg_act_path):
            neg_activations = extract_layer_16_activations(row['neg_system_prompt'], row['question'], row['neg_answer'], model, tokenizer)
            torch.save(neg_activations, neg_act_path)

    print("Activation extraction complete.")

    # 5. Final Vector Computation
    print("\nComputing final hallucination vector...")
    
    def load_and_average_activations(directory):
        all_tensors = [torch.load(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.pt')]
        if not all_tensors:
            return None
        stacked_tensors = torch.stack(all_tensors)
        return stacked_tensors.mean(dim=0)

    mean_pos_activations = load_and_average_activations(pos_dir)
    mean_neg_activations = load_and_average_activations(neg_dir)

    if mean_pos_activations is None or mean_neg_activations is None:
        print("ERROR: Could not load activations for vector computation. Exiting.")
        return

    v_halluc = mean_neg_activations - mean_pos_activations
    torch.save(v_halluc, config.VECTOR_PATH)

    print(f"Hallucination vector computed. Shape: {v_halluc.shape}")
    print(f"Vector saved to: {config.VECTOR_PATH}")
    print("\n--- Step 1 Complete ---")

if __name__ == "__main__":
    main()