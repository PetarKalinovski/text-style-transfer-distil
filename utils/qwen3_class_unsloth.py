from unsloth import FastLanguageModel
import torch
import os
from transformers import TrainingArguments
from trl import SFTTrainer
import mlflow
import mlflow.pytorch
import json
from datasets import Dataset
from pathlib import Path


class QwenModelUnsloth:
    def __init__(self, max_seq_length=1024, load_in_4bit=True):
        self.model_name = "Qwen/Qwen3-4B"
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None

        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        """
        Load the model and tokenizer using Unsloth for memory efficiency.
        """
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit=self.load_in_4bit,  # Use 4bit quantization to reduce memory usage
            # token="hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",     # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,   # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

        return model, tokenizer

    def outputing(self, content):
        """
        Generate a response from the model based on the provided content.
        """
        messages = [{"role": "user", "content": content}]

        # Use FastLanguageModel for inference
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=500,
            temperature=0.8,
            top_p=0.8,
            top_k=20,
            use_cache=True
        )

        # Decode only the new tokens
        response = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return response

    def prepare_dataset_from_jsonl(self, jsonl_file_path):
        """
        Prepare dataset from JSONL file for SFTTrainer
        """
        data = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())

                # Format as conversation
                messages = [
                    {"role": "user", "content": entry["prompt"]},
                    {"role": "assistant", "content": entry["response"]}
                ]

                # Convert to chat format
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                data.append({"text": text})

        return Dataset.from_list(data)

    def finetuning(self, train_dataset, eval_dataset=None, experiment_name="qwen_unsloth_finetuning"):
        """
        Fine-tune the model using Unsloth's optimized training.
        """
        # Set up MLflow
        mlflow_dir = Path.cwd() / "mlruns"
        mlflow_dir.mkdir(exist_ok=True)

        # Convert to URI properly
        mlflow_uri = mlflow_dir.as_uri()
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "model_name": self.model_name,
                "max_seq_length": self.max_seq_length,
                "load_in_4bit": self.load_in_4bit,
                "lora_r": 16,
                "lora_alpha": 16,
                "train_dataset_size": len(train_dataset),
                "eval_dataset_size": len(eval_dataset) if eval_dataset else 0,
            })

            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                dataset_num_proc=2,
                packing=False,  # Can make training 5x faster for short sequences.
                args=TrainingArguments(
                    per_device_train_batch_size=2,  # Reduced batch size for 8GB VRAM
                    per_device_eval_batch_size=2,
                    gradient_accumulation_steps=4,   # Effective batch size = 2 * 4 = 8
                    warmup_steps=100,               # Reduced warmup steps
                    max_steps=500,                  # Use max_steps instead of epochs for shorter training
                    learning_rate=2e-4,             # Higher LR for LoRA
                    fp16=not torch.cuda.is_bf16_supported(),
                    bf16=torch.cuda.is_bf16_supported(),
                    logging_steps=10,
                    optim="adamw_8bit",             # 8-bit optimizer to save memory
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir="./results",
                    logging_dir="./logs",
                    eval_strategy="steps" if eval_dataset else "no",
                    eval_steps=100 if eval_dataset else None,
                    save_steps=100,
                    save_total_limit=2,
                    report_to=["mlflow"],
                    run_name=f"qwen_unsloth_{mlflow.active_run().info.run_id}",
                ),
            )

            # Print memory usage before training
            if torch.cuda.is_available():
                print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

            # Start training
            trainer_stats = trainer.train()

            # Log training results
            mlflow.log_metrics({
                "final_train_loss": trainer_stats.training_loss,
                "train_runtime": trainer_stats.metrics.get("train_runtime", 0),
                "train_samples_per_second": trainer_stats.metrics.get("train_samples_per_second", 0),
            })

            # Save the LoRA adapters
            trainer.model.save_pretrained("./lora_model")
            trainer.tokenizer.save_pretrained("./lora_model")

            # Log the LoRA adapters as artifacts
            mlflow.log_artifacts("./lora_model", artifact_path="lora_adapters")

            print(f"Training completed! LoRA adapters saved to './lora_model'")
            print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    def finetuning_from_jsonl(self, train_jsonl_path, eval_jsonl_path=None, experiment_name="qwen_unsloth_finetuning"):
        """
        Fine-tune directly from JSONL files using Unsloth
        """
        train_dataset = self.prepare_dataset_from_jsonl(train_jsonl_path)
        eval_dataset = None
        if eval_jsonl_path:
            eval_dataset = self.prepare_dataset_from_jsonl(eval_jsonl_path)

        print(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            print(f"Evaluation dataset size: {len(eval_dataset)}")

        self.finetuning(train_dataset, eval_dataset, experiment_name)

    def load_lora_adapters(self, adapter_path="./lora_model"):
        """
        Load fine-tuned LoRA adapters for inference
        """
        from peft import PeftModel

        # Load the base model first
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )

        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, adapter_path)

        self.model = model
        self.tokenizer = tokenizer

        print(f"LoRA adapters loaded from {adapter_path}")

    def print_memory_usage(self):
        """
        Print current GPU memory usage
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")