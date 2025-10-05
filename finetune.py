'''
https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/masked_language_modeling.ipynb#scrollTo=okNHDgXflcrL
This finetuning script is taken from the above colab notebook
'''

from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, TrainerCallback
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset
import torch
import math
import os
from datetime import datetime

logs_base = "./logs/runs"
os.makedirs(logs_base, exist_ok=True)

timestamp = datetime.now().strftime("%m%d_%H%M")
run_name = f"run_{timestamp}"
logging_dir = os.path.join(logs_base, run_name)







tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

block_size = 128

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


'''
Load dataset
split 80/20 train/test
flatten because text is nested in answer
'''
eli5 = load_dataset("dany0407/eli5_category", split="train")
eli5 = eli5.train_test_split(test_size=0.2)
eli5 = eli5.flatten()


# preprocess the dataset 
tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4, # number of processes to use, add more if you can handle it
    remove_columns=eli5["train"].column_names,
)

# group texts into chunks of block_size and create a batch of examples
lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

tokenizer.pad_token = tokenizer.sep_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


# training

class TensorBoardCallback(TensorBoardCallback):
    """Custom TensorBoard callback that adds perplexity metrics."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add perplexity before TensorBoard logging."""
        if logs is not None:
            if "loss" in logs:
                try:
                    logs["train_perplexity"] = math.exp(logs["loss"])
                except (OverflowError, ValueError):
                    logs["train_perplexity"] = float("inf")

            if "eval_loss" in logs:
                try:
                    logs["eval_perplexity"] = math.exp(logs["eval_loss"])
                except (OverflowError, ValueError):
                    logs["eval_perplexity"] = float("inf")

        # Call parent to actually log to TensorBoard
        super().on_log(args, state, control, logs=logs, **kwargs)


training_args = TrainingArguments(
    output_dir="./outputs/distilbert-finetuned-eli5-mlm",
    eval_strategy="steps",
    eval_steps=1000,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=logging_dir,
    report_to=["tensorboard"],
    logging_steps=20,
    save_strategy="steps",
    save_steps=2000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_perplexity",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[TensorBoardCallback()],
)

trainer.train()

# Save the best model
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)

