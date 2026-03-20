"""
farm_gpu/model_training.py

Fine-tunes ai4bharat/indic-bert on the KCC query-type classification task
using the HuggingFace Trainer API.

Usage:
    from farm_gpu.model_training import train
    trainer, tokenized_dataset = train(df)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from config.settings import (
    INDICBERT_MODEL_NAME,
    INDICBERT_OUTPUT_DIR,
    INDICBERT_NUM_LABELS,
    INDICBERT_MAX_LENGTH,
    INDICBERT_TRAIN_EPOCHS,
    INDICBERT_BATCH_SIZE,
    INDICBERT_TEST_SIZE,
    HF_TOKEN,
)
from shared.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class LogEpochCallback(TrainerCallback):
    """Prints a summary line at the start and end of every epoch."""

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n🚀 Starting Epoch {int(state.epoch) + 1} / {int(args.num_train_epochs)}")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"✅ Finished Epoch {int(state.epoch)}\n")


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def build_tokenized_dataset(df, tokenizer) -> DatasetDict:
    """
    Convert a cleaned DataFrame into a tokenised HuggingFace DatasetDict
    with 'train' and 'test' splits.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'Query' and 'label'.
    tokenizer : PreTrainedTokenizer

    Returns
    -------
    DatasetDict
    """
    df = df.copy()
    df["label"] = df["label"].astype("category").cat.codes

    dataset = Dataset.from_pandas(df[["Query", "label"]])

    def tokenize_function(example):
        return tokenizer(
            example["Query"],
            truncation=True,
            padding="max_length",
            max_length=INDICBERT_MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_function, batched=True)
    split = tokenized.train_test_split(test_size=INDICBERT_TEST_SIZE, shuffle=True)

    # Clean up index columns added by to_pandas() round-trips
    def _drop_index_col(ds):
        if "__index_level_0__" in ds.column_names:
            ds = ds.remove_columns(["__index_level_0__"])
        return ds

    return DatasetDict(
        train=_drop_index_col(split["train"]),
        test=_drop_index_col(split["test"]),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df):
    """
    Full training pipeline: tokenise → build model → train → save.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with 'Query' and 'label' columns.

    Returns
    -------
    trainer : Trainer
    tokenized_dataset : DatasetDict
    """
    # Load tokenizer & model
    logger.info("Loading tokenizer and model: %s", INDICBERT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        INDICBERT_MODEL_NAME, num_labels=INDICBERT_NUM_LABELS
    )

    # Prepare data
    tokenized_dataset = build_tokenized_dataset(df, tokenizer)
    logger.info("Dataset sizes – train: %d, test: %d",
                len(tokenized_dataset["train"]), len(tokenized_dataset["test"]))

    # Metrics
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=INDICBERT_OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        num_train_epochs=INDICBERT_TRAIN_EPOCHS,
        per_device_train_batch_size=INDICBERT_BATCH_SIZE,
        per_device_eval_batch_size=INDICBERT_BATCH_SIZE,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(LogEpochCallback())

    logger.info("Starting training …")
    trainer.train()

    # Save locally
    model.save_pretrained(INDICBERT_OUTPUT_DIR)
    tokenizer.save_pretrained(INDICBERT_OUTPUT_DIR)
    logger.info("✅ Model and tokenizer saved to %s", INDICBERT_OUTPUT_DIR)

    return trainer, tokenized_dataset


def push_to_hub(model_repo: str, tokenizer_repo: str):
    """Push the saved model and tokenizer to HuggingFace Hub."""
    from huggingface_hub import login
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    login(HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(INDICBERT_OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(INDICBERT_OUTPUT_DIR)
    model.push_to_hub(model_repo)
    tokenizer.push_to_hub(tokenizer_repo)
    logger.info("Pushed to Hub: %s | %s", model_repo, tokenizer_repo)


if __name__ == "__main__":
    from farm_gpu.data_extraction import fetch_kcc_data
    from farm_gpu.data_preprocessing import build_combined_dataset

    state_dfs = fetch_kcc_data()
    df = build_combined_dataset(state_dfs)
    train(df)
