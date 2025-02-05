import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, pipeline

class MarketReportDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer, max_length: int = 512):
        """
        Loads a CSV with two columns: 'prompt' and 'report'.
        Each row represents one training example.
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the prompt and report to strings to ensure proper tokenization.
        prompt = str(self.data.loc[idx, "prompt"])
        report = str(self.data.loc[idx, "report"])

        # Tokenize the prompt (input) and the report (the desired output)
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        targets = self.tokenizer(
            report,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Squeeze to remove batch dimension
        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        labels = targets.input_ids.squeeze()

        # Replace pad token ids in labels with -100 to ignore loss on padding tokens.
        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def main():
    # Choose your base model; using FLAN-T5 base as a starting point.
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load your fine-tuning CSV file.
    # The CSV should have at least two columns: "prompt" and "report".
    dataset = MarketReportDataset("market_analysis_report_fine_tune.csv", tokenizer, max_length=512)

    training_args = TrainingArguments(
        output_dir="./flan_t5_finetuned_market",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # or even 1 if needed
        learning_rate=5e-5,
        weight_decay=0.01,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="no",
        fp16=True if torch.cuda.is_available() else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Fine-tune the model.
    trainer.train()

    # Save the fine-tuned model and tokenizer.
    model.save_pretrained("./flan_t5_finetuned_market")
    tokenizer.save_pretrained("./flan_t5_finetuned_market")
    print("Fine-tuning completed and model saved to ./flan_t5_finetuned_market.")

if __name__ == "__main__":
    main() 