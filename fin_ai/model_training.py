"""
fin_ai/model_training.py

Fine-tunes bert-base-uncased on the Bitext banking intent-classification
task using the HuggingFace Trainer API.

Usage:
    from fin_ai.model_training import train
    trainer = train(tokenized_dataset, label2id)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import evaluate
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import DatasetDict

from config.settings import (
    BERT_MODEL_NAME,
    BERT_OUTPUT_DIR,
    BERT_TRAIN_EPOCHS,
    BERT_BATCH_SIZE,
)
from shared.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class LogEpochCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n🚀 Starting Epoch {int(state.epoch) + 1} / {int(args.num_train_epochs)}")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"✅ Finished Epoch {int(state.epoch)}\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(tokenized_dataset: DatasetDict, label2id: dict) -> Trainer:
    """
    Fine-tune bert-base-uncased on tokenized_dataset.

    Parameters
    ----------
    tokenized_dataset : DatasetDict
        With 'train' and 'test' splits (from data_preprocessing).
    label2id : dict[int, str]
        Maps class index → intent string label.

    Returns
    -------
    Trainer
    """
    num_labels = len(label2id)
    logger.info("Training BERT with %d labels …", num_labels)

    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=num_labels
    )
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=BERT_OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        num_train_epochs=BERT_TRAIN_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
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

    model.save_pretrained(BERT_OUTPUT_DIR)
    tokenizer.save_pretrained(BERT_OUTPUT_DIR)
    logger.info("✅ Model and tokenizer saved to %s", BERT_OUTPUT_DIR)

    return trainer


if __name__ == "__main__":
    from fin_ai.data_extraction import load_banking_dataset
    from fin_ai.data_preprocessing import build_tokenized_dataset

    df = load_banking_dataset()
    tokenized_dataset, label2id, id2label = build_tokenized_dataset(df)
    train(tokenized_dataset, label2id)
