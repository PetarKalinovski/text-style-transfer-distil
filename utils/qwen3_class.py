from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from transformers import Trainer, TrainingArguments


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

    def finetuning(self, train_dataset, eval_dataset):
        """
        Fine-tune the model using the provided training and evaluation datasets.
        Args:
            train_dataset: The dataset used for training.
            eval_dataset: The dataset used for evaluation.
        """

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
            save_steps=500,
            eval_steps=500,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()