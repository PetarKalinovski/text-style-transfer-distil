import mlflow
import pandas as pd
import json
import mlflow
import numpy as np
from loguru import logger
import torch
import bert_score
import evaluate
from sklearn.metrics import accuracy_score
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.inference import load_merged_model
from utils.qwen3_class import QwenModel

def calculate_perplexity(model, tokenizer, texts):
    """
    Calculate perplexity for a list of texts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of texts to calculate perplexity for

    Returns:
        float: Average perplexity across all texts
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            # Tokenize
            encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = encodings.input_ids.to(model.device)

            # Get model outputs
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            # Accumulate loss and token count
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def combine_datasets(training_subset_with_style2, inference_results, output_path):
    # Load the datasets
    subset_df = pd.read_csv(training_subset_with_style2)
    with open(inference_results, 'r', encoding='utf-8') as f:
        inference_data = json.load(f)

    inference_df = pd.DataFrame(inference_data)

    # Merge the datasets on 'style1'
    combined_df = pd.merge(inference_df, subset_df, left_on='input', right_on='style1', how='left')

    # Save the combined dataset
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Combined dataset saved to {output_path}")

def start_run(dataset_path: str = 'results/combined_dataset.csv', model=None, tokenizer=None):
    # Load dataset

    mlflow.set_tracking_uri("/mlruns")

    dataset = pd.read_csv(dataset_path)

    X_samples = dataset['style1'].tolist()

    y_samples = dataset['style2'].tolist()
    predictions = dataset['transformed'].tolist()
    predictions_no_reasoning = []
    for prediction in predictions:
        if "Transformed Text:" in prediction:
            pred=prediction.split("Transformed Text:")[-1].strip()
        else:
            pred=prediction.split("Transformed Text")[-1].strip()

        pred=pred.lower()
        predictions_no_reasoning.append(pred)


    eval_df = pd.DataFrame({
        "inputs": X_samples,
        "targets": y_samples,
        "predictions": predictions_no_reasoning
    })
    # Compute BERTScore

    P, R, F1 = bert_score.score(
        cands=predictions_no_reasoning,
        refs=y_samples,
        lang='en',
        rescale_with_baseline=True,
        verbose=True
    )

    logger.info(f"BERTScore Precision: {P.mean().item():.4f}")
    logger.info(f"BERTScore Recall: {R.mean().item():.4f}")
    logger.info(f"BERTScore F1: {F1.mean().item():.4f}")

    references = [[target] for target in eval_df['targets'].tolist()]

    bleu_scorer = evaluate.load("bleu")
    bleu_scores = bleu_scorer.compute(predictions=eval_df['predictions'].tolist(),
                                          references=references)


    logger.info(f"BLEU Scores: {bleu_scores}")

    if  model is not None and tokenizer is not None:
        # logger.info("Calculating perplexity on predictions...")
        # pred_perplexity = calculate_perplexity(model, tokenizer, predictions_no_reasoning)
        # logger.info(f"Prediction Perplexity: {pred_perplexity:.4f}")

        logger.info("Calculating perplexity on targets...")
        target_perplexity = calculate_perplexity(model, tokenizer, y_samples)
        logger.info(f"Target Perplexity: {target_perplexity:.4f}")
    else:
        logger.info("Skipping perplexity calculation (model/tokenizer not provided)")



if __name__ == "__main__":
    #model, tokenizer = load_merged_model()
    qwen_class= QwenModel()
    model=qwen_class.model
    tokenizer=qwen_class.tokenizer


    combine_datasets('data/training_subset_with_style2.csv', 'results/qwen3_base_inference_style_transfer.json', 'results/combined_dataset.csv')


    start_run(model=model, tokenizer=tokenizer)