# 2_train_risk_classifier.py
"""
Step 2: Train the Prompt-Risk Classifier.

This script replicates the logic from the second notebook:
1.  Loads the pre-computed hallucination vector (v_halluc.pt).
2.  Loads the SQuAD dataset and creates a sample of 2000 prompts.
3.  Generates three scenarios for each prompt (standard, no-context, distractor)
    to elicit a mix of correct and hallucinated responses.
4.  Runs a resilient generation and judging loop to create a labeled dataset, where a
    specialized judge determines if a response is a hallucination (1) or not (0).
5.  Runs a resilient feature calculation loop to compute the 'z_feature' for each prompt
    (the dot product of its activation with v_halluc).
6.  Trains a logistic regression classifier on the (z_feature, hallucination_label) data.
7.  Evaluates the classifier's performance on a test set, saving the model, AUROC score,
    and a plot of the ROC curve.
"""
import os
import pandas as pd
import numpy as np
import torch
import joblib
import json
import requests
import re
from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import config

import config
import utils

def judge_squad_answer(context, question, model_answer, ground_truth_answer, api_key):
    """Gets a binary hallucination label (1 for YES, 0 for NO) from the LLM judge."""
    prompt = f"""
You are an impartial judge evaluating a language model's answer for its faithfulness to a given context.
Your task is to determine if the model's answer contains any fabricated information not supported by the context.

Here is the provided context:
<context>
{context or "No context was provided."}
</context>

Here is the question that was asked:
<question>
{question}
</question>

Here is the ground-truth answer, which is known to be correct and derived from the context:
<ground_truth_answer>
{ground_truth_answer}
</ground_truth_answer>

Here is the language model's generated answer:
<model_answer>
{model_answer}
</model_answer>

**Your Task:**
Carefully compare the <model_answer> against the <context>. Does the <model_answer> contain any facts, details, or claims that are NOT supported by the <context>? Ignore minor differences in phrasing. If the model says it cannot find the answer, that is not a hallucination.

Respond with only the word 'YES' if the model's answer contains fabricated information, or 'NO' if it is faithful to the context.
"""
    url = "https://api.scaledown.xyz/compress/"
    payload = json.dumps({
        "context": "You are an impartial judge evaluating for hallucinations.",
        "prompt": prompt, "model": config.LLM_JUDGE_MODEL
    })
    headers = {'x-api-key': api_key, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=60)
        response.raise_for_status()
        content = response.json().get("full_response", "").strip().upper()
        if 'YES' in content: return 1
        if 'NO' in content: return 0
        return -1
    except Exception:
        return -1

def generate_squad_answer_multi_scenario(model, tokenizer, row):
    """Generates an answer based on the scenario specified in the DataFrame row."""
    SYSTEM_PROMPT = "You are a helpful assistant. Answer the following question based ONLY on the provided context."
    scenario, question = row['scenario'], row['question']
    context_to_use = ""
    if scenario == 'standard':
        context_to_use = row['context']
    elif scenario == 'distractor':
        context_to_use = row['distractor_context']
    
    user_prompt = f"Context:\n{context_to_use}\n\nQuestion:\n{question}" if context_to_use else f"Question:\n{question}"
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    full_prompt_templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(full_prompt_templated, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response_text.strip(), full_prompt_templated

def main():
    """Main execution function for training the risk classifier."""
    print("--- Step 2: Training the Prompt-Risk Classifier ---")

    # 1. Setup
    secrets = utils.load_secrets()
    api_key = secrets.get('SCALEDOWN_API_KEY')
    if not api_key:
        print("ERROR: SCALEDOWN_API_KEY not found. Cannot proceed. Exiting.")
        return

    model, tokenizer = utils.load_model_and_tokenizer()

    if not os.path.exists(config.VECTOR_PATH):
        print(f"ERROR: Hallucination vector not found at {config.VECTOR_PATH}. Please run Step 1 first. Exiting.")
        return
    v_halluc = torch.load(config.VECTOR_PATH).to(model.device)
    print("Hallucination vector loaded successfully.")

    # 2. Dataset Generation and Labeling
    if not os.path.exists(config.SQUAD_LABELED_ANSWERS_PATH):
        print("\nLabeled SQuAD dataset not found. Starting generation and labeling process...")
        squad_dataset = load_dataset(config.SQUAD_DATASET_NAME, split="train")
        sampled_df = squad_dataset.to_pandas().sample(n=config.SQUAD_SAMPLES, random_state=42).reset_index(drop=True)
        
        # Prepare scenarios
        distractor_indices = np.random.permutation(sampled_df.index)
        sampled_df['distractor_context'] = sampled_df.loc[distractor_indices, 'context'].values
        scenarios = (['standard'] * 1000) + (['no_context'] * 500) + (['distractor'] * 500)
        sampled_df['scenario'] = scenarios
        
        results_list = []
        with tqdm(total=len(sampled_df), desc="Generating & Judging SQuAD") as pbar:
            for i, row in sampled_df.iterrows():
                model_answer, full_prompt = generate_squad_answer_multi_scenario(model, tokenizer, row)
                ground_truth_answer = row['answers']['text'][0] if row['answers']['text'] else ""
                label = judge_squad_answer(row['context'], row['question'], model_answer, ground_truth_answer, api_key)
                
                results_list.append({
                    'scenario': row['scenario'], 'full_prompt': full_prompt, 'model_answer': model_answer,
                    'ground_truth_answer': ground_truth_answer, 'hallucination_label': label,
                    'original_context': row['context'], 'question': row['question']
                })
                
                if (i + 1) % config.BATCH_SIZE_GENERATE == 0:
                    pd.DataFrame(results_list).to_csv(config.SQUAD_LABELED_ANSWERS_PATH, index=False)
                
                pbar.update(1)
        pd.DataFrame(results_list).to_csv(config.SQUAD_LABELED_ANSWERS_PATH, index=False)
        print(f"Labeled dataset saved to {config.SQUAD_LABELED_ANSWERS_PATH}")
    else:
        print(f"\nFound existing labeled dataset at {config.SQUAD_LABELED_ANSWERS_PATH}. Skipping generation.")

    # 3. Feature Calculation
    print("\nStarting feature calculation...")
    labeled_df = pd.read_csv(config.SQUAD_LABELED_ANSWERS_PATH)
    labeled_df = labeled_df[labeled_df['hallucination_label'] != -1].dropna(subset=['full_prompt'])
    labeled_df['hallucination_label'] = labeled_df['hallucination_label'].astype(int)

    if os.path.exists(config.SQUAD_FEATURES_PATH):
        df_to_process = pd.read_csv(config.SQUAD_FEATURES_PATH)
        start_index = df_to_process['z_feature'].notna().sum()
        print(f"Resuming feature calculation from index {start_index}.")
    else:
        df_to_process = labeled_df.copy()
        df_to_process['z_feature'] = np.nan
        start_index = 0
        print("Starting feature calculation from scratch.")
    
    with tqdm(total=len(df_to_process), initial=start_index, desc="Calculating z-features") as pbar:
        for i in range(start_index, len(df_to_process)):
            if pd.notna(df_to_process.loc[i, 'z_feature']):
                continue
            
            prompt = df_to_process.loc[i, 'full_prompt']
            activation = utils.get_last_prompt_token_activation(model, tokenizer, prompt)
            projection = torch.dot(activation.to(v_halluc.dtype), v_halluc)
            df_to_process.loc[i, 'z_feature'] = projection.item()
            
            if (i + 1) % config.BATCH_SIZE_FEATURES == 0:
                df_to_process.to_csv(config.SQUAD_FEATURES_PATH, index=False)
            pbar.update(1)

    df_to_process.to_csv(config.SQUAD_FEATURES_PATH, index=False)
    print(f"Feature calculation complete. Final dataset saved to {config.SQUAD_FEATURES_PATH}")

    # 4. Train and Evaluate Classifier
    print("\nTraining and evaluating the logistic regression classifier...")
    final_df = pd.read_csv(config.SQUAD_FEATURES_PATH).dropna(subset=['z_feature'])
    
    X = final_df[['z_feature']]
    y = final_df['hallucination_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    risk_classifier = LogisticRegression(random_state=42)
    risk_classifier.fit(X_train, y_train)
    print("Training complete.")
    
    joblib.dump(risk_classifier, config.CLASSIFIER_PATH)
    print(f"Classifier saved to: {config.CLASSIFIER_PATH}")
    
    # 5. Evaluate Performance
    y_pred_proba = risk_classifier.predict_proba(X_test)[:, 1]
    auroc_score = roc_auc_score(y_test, y_pred_proba)
    
    print("\n--- Performance Evaluation ---")
    print(f"AUROC Score on Test Set: {auroc_score:.4f}")
    if auroc_score >= 0.75:
        print("Success! AUROC meets or exceeds the target of 0.75.")
    else:
        print("Warning: AUROC is below the target of 0.75.")
        
    # Plot and save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(config.ROC_CURVE_PLOT_PATH)
    print(f"ROC curve plot saved to: {config.ROC_CURVE_PLOT_PATH}")
    
    print("\nClassification Report (at 0.5 threshold):")
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    print(classification_report(y_test, y_pred_binary, target_names=['Faithful (0)', 'Hallucination (1)']))
    
    print("\n--- Step 2 Complete ---")

if __name__ == "__main__":
    main()