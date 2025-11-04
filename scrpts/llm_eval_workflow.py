import asyncio
import json
from pathlib import Path
import sys
import os
import re
import warnings
from dotenv import load_dotenv
from litellm import acompletion
from tqdm.asyncio import tqdm
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands_tools import workflow
import platform

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

# Suppress SSL cleanup warnings on Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    warnings.filterwarnings('ignore', category=ResourceWarning)



def evaluate_single_item(item: dict) -> dict:
    """Evaluate a single item with all metrics in parallel."""
    original_text = item['input']
    prediction = item['transformed']

    # Clean transformed text
    if "Transformed Text:" in prediction:
        transformed_text = prediction.split("Transformed Text:")[-1].strip()
    else:
        transformed_text = prediction.split("Transformed Text")[-1].strip()

    prompts = {
        'preservation': f"""You are evaluating content preservation in text style transfer.

Original Text: "{original_text}"
Transformed Text: "{transformed_text}"

Rate from 1-10 how well the transformed text preserves the factual content and core meaning of the original:
- 10: All key information preserved, no meaning lost
- 7-9: Minor details missing but main points intact
- 4-6: Some important information lost or altered
- 1-3: Significant meaning changed or lost

Return ONLY a single number (1-10).""",

        'accuracy': f"""You are evaluating style neutralization quality.

Original Text: "{original_text}"
Transformed Text: "{transformed_text}"

The goal is neutral, unbiased text. Rate from 1-10 how successfully biased/subjective language was removed:
- 10: Completely neutral, all bias removed, objective tone
- 7-9: Mostly neutral with minor subjective traces
- 4-6: Some bias remains, partially neutral
- 1-3: Still heavily biased or subjective

Look for: emotional language, loaded terms, opinion words, one-sided framing.

Return ONLY a single number (1-10).""",

        'fluency': f"""You are evaluating text fluency and readability.

Transformed Text: "{transformed_text}"

Rate from 1-10 how fluent and natural the text reads:
- 10: Perfect grammar, natural flow, reads like native writing
- 7-9: Minor awkwardness but clear and readable
- 4-6: Noticeable grammar issues or unnatural phrasing
- 1-3: Broken grammar, hard to understand, very awkward

Consider: grammar correctness, sentence structure, word choice, overall readability.

Return ONLY a single number (1-10).""",

        'reasoning': f"""You are evaluating the quality of explanations for style transfer changes.

Original Text: "{original_text}"
Transformed Text: "{prediction}"

Rate from 1-10 how well the reasoning explains the changes:
- 10: Every change clearly justified with specific explanations
- 7-9: Most changes explained, minor gaps
- 4-6: Vague or incomplete explanations
- 1-3: Poor or missing justification

Return ONLY a single number (1-10)."""
    }
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = LiteLLMModel(
        client_args={
            "api_key": api_key,
        },
        model_id="openrouter/google/gemini-2.5-flash",
        params={
            "max_tokens": 5000,
            "temperature": 0.7,
        }
    )
    agent=Agent(model=model, tools=[workflow])

    print("Creating workflow for LLM evaluation...")

    result = agent.tool.workflow(
        action="create",
        workflow_id="llm_evaluation_workflow",
        tasks=[
            {
                "task_id": metric,
                "description": "Evaluate the following prompt and return a score from 1 to 10.",
                "system_prompt": prompt,
                "model_provider": "litellm",
                "model_settings": {
                    "model_id": "openrouter/google/gemini-2.5-flash",
                    "params": {
                        "max_tokens": 500,
                        "temperature": 0.2,
                    }
                }
            }
            for metric, prompt in prompts.items()
        ]
    )

    print("Starting workflow execution...")

    result= agent.tool.workflow(
        action="start",
        workflow_id="llm_evaluation_workflow",
    )

    print("Checking workflow status...")

    result= agent.tool.workflow(
        action="status",
        workflow_id="llm_evaluation_workflow"
    )

    print("Retrieving workflow results...")
    print(result)

    result = agent.tool.workflow(action="list")
    print(result)



    # return {
    #     'original': original_text,
    #     'transformed': transformed_text,
    #     'scores': scores,
    #     'raw_responses': responses
    # }


def llm_judge_on_data(eval_data_path: str) -> list:
    """
    Use LLM to judge the quality of style transfer on the eval data.
    Args:
        eval_data_path: Path to the evaluation data JSON file
    Returns:
        list of dict: Evaluation results with LLM judgments
    """
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    print(f"Starting evaluation of {len(eval_data)} items...")

    results = []
    evaluate_single_item(eval_data[0])
    # for item in eval_data:
    #     eval_result = evaluate_single_item(item)
    #     results.append(eval_result)
    #
    # output_path = eval_data_path.replace('.json', '_judged.json')
    #
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)
    #
    # print(f"Evaluation complete. Results saved to {output_path}")
