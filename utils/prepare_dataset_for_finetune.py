import json
from datasets import Dataset
from transformers import AutoTokenizer

def prepare_style_transfer_dataset(jsonl_file_path, tokenizer, max_length=1024):
    """
    Prepare dataset from your JSONL file for fine-tuning Qwen model

    Args:
        jsonl_file_path: Path to your JSONL file
        tokenizer: The Qwen tokenizer
        max_length: Maximum sequence length

    Returns:
        Dataset: Prepared dataset for training
    """

    # Load the data
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)

    # Format the data for chat-based training
    formatted_data = []

    for entry in data:
        # Create a conversation format
        messages = [
            {
                "role": "user",
                "content": entry["prompt"]
            },
            {
                "role": "assistant",
                "content": entry["response"]
            }
        ]

        formatted_data.append({"messages": messages})

    # Convert to Dataset
    dataset = Dataset.from_list(formatted_data)

    def tokenize_function(examples):
        formatted_texts = []

        for messages in examples["messages"]:
            # Use Qwen's chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  # Don't add generation prompt for training
            )
            formatted_texts.append(text)

        # Tokenize
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            padding=False,  # We'll pad in the data collator
            max_length=max_length,
            return_tensors=None
        )

        # For causal LM training, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names  # Remove original columns
    )

    return tokenized_dataset