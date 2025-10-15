from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from transformers import Trainer, TrainingArguments
import mlflow
import mlflow.pytorch
from transformers.integrations import MLflowCallback
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.prepare_dataset_for_finetune import prepare_style_transfer_dataset


class QwenModel:
    def __init__(self):
        self.model_name = "Qwen/Qwen3-4B"
        self.model = None
        self.tokenizer = None
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

        self.model, self.tokenizer = self.model_and_tokenizer()

    def model_and_tokenizer(self):
        """
        Load the model and tokenizer from the specified model name.

        Returns:
            model: The loaded language model.
            tokenizer: The corresponding tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, cache_dir=self.cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir=self.cache_dir,
            load_in_8bit=True,  # Enabling 4-bit quantization
        )
        return model, tokenizer

    def outputing(self, content):
        """
        Generate a response from the model based on the provided content.
        Args:
            content (str): The input content to prompt the model.
        Returns:
            output: The generated output from the model.
        """
        messages = [{"role": "user", "content": content}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # Must add for generation
            enable_thinking=True,  # Disable thinking
        )

        output = self.model.generate(
            **self.tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=500,
            temperature=0.8,
            top_p=0.8,
            top_k=20,
        )
        return output

    def finetuning(self, train_dataset, eval_dataset, experiment_name="cot_finetuning"):
        """
        Fine-tune the model using the provided training and evaluation datasets.
        Args:
            train_dataset: The dataset used for training.
            eval_dataset: The dataset used for evaluation.
            experiment_name: Name of the MLflow experiment.
        """

        # Set up MLflow with local file tracking (creates mlruns and mlartifacts locally)
        mlflow.set_tracking_uri(".")
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log model and training parameters
            mlflow.log_params({
                "model_name": self.model_name,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "warmup_steps": 500,
                "weight_decay": 0.01,
                "learning_rate": 5e-5,  # Default from TrainingArguments
                "train_dataset_size": len(train_dataset) if train_dataset else 0,
                "eval_dataset_size": len(eval_dataset) if eval_dataset else 0,
            })

            # Log model info
            mlflow.log_params({
                "torch_dtype": str(torch.bfloat16),
                "device_map": "cuda",
                "cache_dir": self.cache_dir,
            })

            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                evaluation_strategy="steps",
                save_total_limit=2,
                save_steps=20,
                eval_steps=20,
                report_to=["mlflow"],  # Enable MLflow integration
                run_name=f"qwen_finetuning_{mlflow.active_run().info.run_id}",
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

            # Start training
            training_result = trainer.train()

            # Log training results
            mlflow.log_metrics({
                "final_train_loss": training_result.training_loss,
                "train_runtime": training_result.metrics.get("train_runtime", 0),
                "train_samples_per_second": training_result.metrics.get("train_samples_per_second", 0),
                "train_steps_per_second": training_result.metrics.get("train_steps_per_second", 0),
            })

            # Log the final model
            mlflow.pytorch.log_model(
                pytorch_model=self.model,
                artifact_path="model",
                registered_model_name="qwen_finetuned_model"
            )

            # Log the tokenizer
            tokenizer_path = "./tokenizer_temp"
            self.tokenizer.save_pretrained(tokenizer_path)
            mlflow.log_artifacts(tokenizer_path, artifact_path="tokenizer")

    def finetuning_from_jsonl(self, train_jsonl_path, eval_jsonl_path=None,
                              experiment_name="style_transfer_finetuning"):
        """
        Fine-tune directly from JSONL files

        Args:
            train_jsonl_path: Path to training JSONL file
            eval_jsonl_path: Path to evaluation JSONL file (optional)
            experiment_name: MLflow experiment name
        """

        # Prepare datasets
        train_dataset = self.prepare_style_transfer_dataset(train_jsonl_path,self.tokenizer)
        eval_dataset = None
        if eval_jsonl_path:
            eval_dataset = self.prepare_style_transfer_dataset(eval_jsonl_path,self.tokenizer)

        print(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            print(f"Evaluation dataset size: {len(eval_dataset)}")

        # Start fine-tuning
        self.finetuning(train_dataset, eval_dataset, experiment_name)