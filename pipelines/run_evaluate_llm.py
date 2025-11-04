from pathlib import Path
import json
from loguru import logger
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scrpts.llm_eval import llm_judge_on_data
from scrpts.llm_eval_workflow import llm_judge_on_data as llm_judge_on_data_workflow


def main():
    eval_data_path = Path.cwd() / "results" / "new_test_results.json"

    logger.info(f"Evaluating LLM judgments on data from {eval_data_path}")

    results=llm_judge_on_data(str(eval_data_path))

    print(json.dumps(results, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    main()