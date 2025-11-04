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
import platform

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

# Suppress SSL cleanup warnings on Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    warnings.filterwarnings('ignore', category=ResourceWarning)


async def get_score(prompt: str, api_key: str, metric_name: str, semaphore: asyncio.Semaphore) -> dict:
    """Get a single score asynchronously with rate limiting."""
    async with semaphore:
        try:
            response = await acompletion(
                model="openrouter/anthropic/claude-sonnet-4.5",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
                api_key=api_key
            )
            text = response.choices[0].message.content.strip()

            # Extract score
            match = re.search(r'\b([1-9]|10)\b', text)
            score = int(match.group(1)) if match else None

            return {"metric": metric_name, "score": score, "response": text}
        except Exception as e:
            print(f"Error getting {metric_name} score: {e}")
            return {"metric": metric_name, "score": None, "error": str(e)}


async def evaluate_single_item(item: dict, api_key: str, semaphore: asyncio.Semaphore) -> dict:
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

    # Run all 4 evaluations in parallel for this item
    tasks = [get_score(prompt, api_key, metric, semaphore) for metric, prompt in prompts.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions in results
    scores = {}
    responses = {}
    for result in results:
        if isinstance(result, dict):
            metric = result['metric']
            scores[metric] = result['score']
            responses[metric] = result.get('response', '')
        else:
            # Handle exception
            print(f"Exception in evaluation: {result}")

    return {
        'original': original_text,
        'transformed': transformed_text,
        'scores': scores,
        'raw_responses': responses
    }


async def llm_judge_on_data_async(eval_data_path: str, max_concurrent: int = 10) -> list:
    """
    Use LLM to judge the quality of style transfer on the eval data asynchronously.
    """
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    output_path = eval_data_path.replace('.json', 'claude_judged.json')

    # Load past results and create efficient lookup
    past_results_dict = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            past_results = json.load(f)
            past_results_dict = {res['original']: res for res in past_results}
        print(f"Loaded {len(past_results_dict)} existing results")

    # Filter items that need evaluation
    items_to_evaluate = [
        item for item in eval_data
        if item['input'] not in past_results_dict
    ]

    print(
        f"Evaluating {len(items_to_evaluate)}/{len(eval_data)} items (skipping {len(eval_data) - len(items_to_evaluate)} cached)...")

    if not items_to_evaluate:
        print("All items already evaluated!")
        return list(past_results_dict.values())

    semaphore = asyncio.Semaphore(max_concurrent)

    # Evaluate only new items
    tasks = [evaluate_single_item(item, api_key, semaphore) for item in items_to_evaluate]
    new_results = []
    completed_count = 0
    save_every = 50

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        result = await coro
        new_results.append(result)
        completed_count += 1

        # Periodic save: merge with existing results
        if completed_count % save_every == 0:
            all_results = list(past_results_dict.values()) + new_results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\nCheckpoint: Saved {len(all_results)} total results")

    # Final save: merge all results
    all_results = list(past_results_dict.values()) + new_results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Completed! Total results: {len(all_results)}")
    return all_results

def llm_judge_on_data(eval_data_path: str, max_concurrent: int = 10) -> list:
    """
    Synchronous wrapper for async evaluation.

    Args:
        eval_data_path: Path to the evaluation data JSON file
        max_concurrent: Maximum number of concurrent API requests

    Returns:
        list of dict: Evaluation results with LLM judgments
    """
    if platform.system() == 'Windows':
        # Use asyncio.run with proper cleanup on Windows
        return asyncio.run(llm_judge_on_data_async(eval_data_path, max_concurrent))
    else:
        # Standard approach for non-Windows
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(llm_judge_on_data_async(eval_data_path, max_concurrent))
        finally:
            loop.close()
