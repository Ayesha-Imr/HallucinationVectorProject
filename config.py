# config.py
"""
Central configuration file for the Hallucination Guardrail project.
This file contains all parameters, file paths, and settings to ensure
that experiments are reproducible and easy to modify without changing the
core logic of the pipeline scripts.
"""

import os

# -----------------------------------------------------------------------------
# EXECUTION ENVIRONMENT
# -----------------------------------------------------------------------------
# Set to 'colab' to use Google Colab's specific features (e.g., userdata for secrets)
# Set to 'local' to use a local environment (e.g., .env file for secrets)
ENVIRONMENT = "local"

# -----------------------------------------------------------------------------
# FILE PATHS
# -----------------------------------------------------------------------------
# For Colab, this assumes your Google Drive is mounted at /content/drive
BASE_DIR = "/content/drive/MyDrive/"
PROJECT_DIR = os.path.join(BASE_DIR, "HallucinationVectorProject")

# Subdirectories for organizing the project
DATA_DIR = os.path.join(PROJECT_DIR, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# --- Input Data Paths ---
HALLUCINATING_TRAIT_DATA_PATH = os.path.join(DATA_DIR, "hallucinating.json")
SQUAD_DATASET_NAME = "squad"
TRUTHFULQA_DATASET_NAME = "domenicrosati/TruthfulQA"


# --- Artifact Paths ---
# Step 1 Outputs
JUDGED_ANSWERS_PATH = os.path.join(RESULTS_DIR, "judged_answers.csv")
VECTOR_PATH = os.path.join(ARTIFACTS_DIR, "v_halluc.pt")

# Step 2 Outputs
SQUAD_LABELED_ANSWERS_PATH = os.path.join(RESULTS_DIR, "squad_labeled_answers_multi_scenario.csv")
SQUAD_FEATURES_PATH = os.path.join(RESULTS_DIR, "squad_data_with_features.csv")
CLASSIFIER_PATH = os.path.join(ARTIFACTS_DIR, "risk_clf.joblib")
ROC_CURVE_PLOT_PATH = os.path.join(PLOTS_DIR, "roc_auc_curve.png")

# Step 3 Outputs
VALIDATION_SET_PATH = os.path.join(DATA_DIR, "validation_set_truthfulqa.csv")
FINAL_TEST_SET_PATH = os.path.join(DATA_DIR, "final_test_set_truthfulqa.csv")
ALPHA_TUNING_RESULTS_PATH = os.path.join(RESULTS_DIR, "alpha_tuning_results.csv")
RISK_THRESHOLDS_PATH = os.path.join(ARTIFACTS_DIR, "risk_thresholds.joblib")
ALPHA_TUNING_PLOT_PATH = os.path.join(PLOTS_DIR, "alpha_tuning_plot.png")

# Step 4 Outputs
GUARDED_RESULTS_PATH = os.path.join(RESULTS_DIR, "final_guarded_results_truthfulqa.csv")
BASELINE_RESULTS_PATH = os.path.join(RESULTS_DIR, "final_baseline_results_truthfulqa.csv")
GUARDED_JUDGED_RESULTS_PATH = os.path.join(RESULTS_DIR, "final_guarded_judged_results.csv")
BASELINE_JUDGED_RESULTS_PATH = os.path.join(RESULTS_DIR, "final_baseline_judged_results.csv")
RISK_COVERAGE_PLOT_PATH = os.path.join(PLOTS_DIR, "risk_coverage_plot.png")
FINAL_SUMMARY_TABLE_PATH = os.path.join(RESULTS_DIR, "final_summary_table.csv")


# -----------------------------------------------------------------------------
# MODEL & TOKENIZER CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
MODEL_DTYPE = None  # Unsloth handles this automatically

# -----------------------------------------------------------------------------
# VECTOR & GUARDRAIL PARAMETERS
# -----------------------------------------------------------------------------
# The transformer layer from which to extract hidden states
TARGET_LAYER = 16

# --- Hyperparameter Tuning ---
# The range of alpha values to test for the steering vector
ALPHA_SEARCH_SPACE = [-3.0, -2.0, -1.0, 0.0] # Simplified based on final results

# The optimal alpha value determined from tuning
OPTIMAL_ALPHA = -3.0

# Percentiles for calculating risk thresholds from the validation set risk scores
TAU_LOW_PERCENTILE = 50
TAU_HIGH_PERCENTILE = 75

# TAU values
TAU_LOW = 0.0109 
TAU_HIGH = 0.0208

# Coherence threshold for selecting optimal alpha
ACCEPTABLE_COHERENCE_THRESHOLD = 80

# --- Vector Building ---
POS_HALLUCINATION_THRESHOLD = 80
NEG_HALLUCINATION_THRESHOLD = 20
COHERENCE_THRESHOLD = 50

# -----------------------------------------------------------------------------
# DATASET & PROCESSING CONFIGURATION
# -----------------------------------------------------------------------------
# Number of samples to draw from SQuAD for training the risk classifier
SQUAD_SAMPLES = 2000

# Number of samples to use for the validation set from TruthfulQA
TRUTHFULQA_VALIDATION_SIZE = 200
TRUTHFULQA_TUNING_SAMPLE_SIZE = 50 # Smaller subset of validation for faster alpha tuning

# Batch sizes for resilient loops to save progress frequently
BATCH_SIZE_GENERATE = 20
BATCH_SIZE_FEATURES = 50
BATCH_SIZE_EVAL = 5

# LLM Judge
LLM_JUDGE_MODEL = "gpt-4o"

# -----------------------------------------------------------------------------
# HUGGING FACE HUB CONFIGURATION
# -----------------------------------------------------------------------------
# Repository for uploading final artifacts and datasets
HF_DATASET_REPO = "scaledown/persona_vectors_project_datasets"
HF_MODEL_REPO = "scaledown/persona_vector_project_artifacts"