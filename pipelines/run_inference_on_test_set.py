from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.inference import run_inference_on_test_set, inferene_on_base_model


def main():
    """
    Run inference on the test set using the fine-tuned model.
    """

   # run_inference_on_test_set("data/testing_subset.csv", "results/test_set_inference_results.json")

    inferene_on_base_model("data/testing_subset.csv", "results/test_set_inference_base_model_results.json")

if __name__ == "__main__":
    main()