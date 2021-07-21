import argparse
import pandas as pd
from rich.progress import track
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

print("Packages loaded")

# Parse the Args
parser = argparse.ArgumentParser(
    description="Finetune Bert Model for Downstream Text Classification task."
)
parser.add_argument("--dev", type=bool, help="Whether or not to train on whole dataset")
parser.add_argument("--epochs", type=int, help="Number of epochs to train model")
args = parser.parse_args()


class MIMICDataset(Dataset):
    def __init__(self, data, max_len=128):
        self.data = data
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        text = str(data_row.TEXT)
        labels = data_row.iloc[2:]

        tokenized_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        ids = tokenized_text["input_ids"].flatten().numpy()
        mask = tokenized_text["attention_mask"].flatten().numpy()
        token_type_ids = tokenized_text["token_type_ids"].flatten().numpy()

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.FloatTensor(labels),
        }


class BERTModel(nn.Module):
    def __init__(self, num_labels):
        super(BERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", return_dict=True
        )
        self.dropout = nn.Dropout(0.3)
        if not args.dev:
            self.hidden = nn.Linear(768, 64)
            self.classifier = nn.Linear(64, num_labels)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, model_dict):
        out = self.bert(
            model_dict["input_ids"], model_dict["attention_mask"], model_dict['token_type_ids'])
        # print(out)
        pooled_output = self.dropout(out.pooler_output)
        if not args.dev:
            pooled_output = self.hidden(pooled_output)
            pooled_output =self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train(model, train_dataloader, valid_dataloader, test_dataloader, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps= 0,
        num_training_steps=args.epochs * len(train_dataloader)
    )

    print("start of training loop")
    for epoch in track(range(args.epochs)):
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        train_f1_macro = 0
        train_f1_micro = 0
        count = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            target = batch["target"].to(device)
            input_dict = {"input_ids": ids, "attention_mask": mask, "token_type_ids":token_type_ids}
            logits = model(input_dict)
            loss = criterion(logits, batch["target"])
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * batch["ids"].size(0)
            wandb.log({"train_loss": loss})

            # Metrics - TODO add various metrics
            probs = torch.sigmoid(logits).cpu().detach()
            predicted = (probs > 0.5).float().numpy()
            actual = target.cpu().numpy()

            train_acc += accuracy_score(actual, predicted)
            train_f1_macro += f1_score(actual, predicted, average='macro')
            train_f1_micro += f1_score(actual, predicted, average='micro')
            wandb.log({{"train_acc": accuracy_score(actual, predicted), 
                        "train_f1_macro": f1_score(actual, predicted, average='macro'),
                        "train_f1_micro": f1_score(actual, predicted, average='micro')}})
            count += 1
        train_loss /= len(train_dataloader.sampler)
        train_acc /= count
        train_f1_macro /= count
        train_f1_micro /= count

        # Validation
        model.eval()
        valid_loss = 0
        valid_acc = 0
        valid_f1_macro = 0
        valid_f1_micro = 0
        count = 0
        for batch in valid_dataloader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            target = batch["target"].to(device)

            input_dict = {"input_ids": ids, "attention_mask": mask, "token_type_ids":token_type_ids}
            logits = model(input_dict)
            loss = criterion(logits, batch["target"])
            valid_loss += loss.item() * batch["ids"].size(0)
            wandb.log({"valid_loss": loss})

            # Metrics - TODO add various metrics
            probs = torch.sigmoid(logits).cpu().detach()
            predicted = (probs > 0.5).float().numpy()
            actual = target.cpu().numpy()

            valid_acc += accuracy_score(actual, predicted)
            valid_f1_macro += f1_score(actual, predicted, average="macro")
            valid_f1_micro += f1_score(actual, predicted, average="micro")
            wandb.log({{"valid_acc": accuracy_score(actual, predicted), 
                        "valid_f1_macro": f1_score(actual, predicted, average='macro'),
                        "valid_f1_micro": f1_score(actual, predicted, average='micro')}})
            count += 1

        valid_loss /= len(valid_dataloader.sampler)
        valid_acc /= count
        valid_f1_macro /= count
        valid_f1_micro /= count

        print(
            "\n"
            "\n"
            f"Epoch {epoch+1}/{args.epochs}.. "
            f"Train loss: {train_loss:.3f}.. "
            f"Validation loss: {valid_loss:.3f}.."
            "\n"
            f"Train accuracy: {train_acc:.3f}"
            f"Validation accuracy: {valid_acc:.3f}"
            "\n"
            f"Train F1 Scores: macro - {train_f1_macro:.3f} micro - {train_f1_micro:.3f}\n"
            f"Valid F1 Scores: macro - {valid_f1_macro:.3f} micro - {valid_f1_micro:.3f}\n"
        )

    # Testing
    model.eval()
    test_loss = 0
    test_acc = 0
    test_f1_macro = 0
    test_f1_micro = 0
    count = 0
    for batch in test_dataloader:
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        target = batch["target"].to(device)

        input_dict = {"input_ids": ids, "attention_mask": mask, "token_type_ids":token_type_ids}
        logits = model(input_dict)

        # Metrics - TODO add various metrics
        probs = torch.sigmoid(logits).cpu().detach()
        predicted = (probs > 0.5).float().numpy()
        actual = target.cpu().numpy()

        test_acc += accuracy_score(actual, predicted)
        test_f1_macro += f1_score(actual, predicted, average="macro")
        test_f1_micro += f1_score(actual, predicted, average="micro")
        count += 1

    test_acc /= count
    test_f1_macro /= count
    test_f1_micro /= count

    test_roc_auc_macro = roc_auc_score(actual, predicted, average="macro")
    test_roc_auc_micro = roc_auc_score(actual, predicted, average="micro")

    test_roc_auc = roc_auc_score(actual, predicted, average=None, labels=labels)

    wandb.log({{"test_acc": test_acc, 
                "test_f1_macro": test_f1_macro,
                "test_f1_micro": test_f1_micro,
                "test_roc_auc_macro": test_roc_auc_macro,
                "test_roc_auc_micro": test_roc_auc_micro,
                "test_roc_auc_by_class": test_roc_auc, }})

    wandb.log({"roc_curve" : wandb.plot.roc_curve(actual,
           predicted, labels=labels)})

    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(
                        y_true=actual,
                        preds=predicted,
                        class_names=labels)})



    print(
        "Training Finished!\n"
        "Final Metrics: \n"
        f"Train loss: {train_loss:.3f} \n"
        f"Validation loss: {valid_loss:.3f}\n"
        "\n"
        f"Train accuracy: {train_acc:.3f}\n"
        f"Validation accuracy: {valid_acc:.3f}\n"
        f"Test accuracy: {test_acc:.3f}"
        "\n"
        f"Train F1 Scores: macro - {train_f1_macro:.3f} micro - {train_f1_micro:.3f}\n"
        f"Valid F1 Scores: macro - {valid_f1_macro:.3f} micro - {train_f1_micro:.3f}\n"
        f"Test  F1 Scores: macro - {test_f1_macro:.3f}  micro - {test_f1_micro:.3f}\n"
        "\n"
        f"Test Macro AUC: {test_roc_auc_macro:.3f}\n"
        f"Test Micro AUC: {test_roc_auc_micro:.3f}\n"
        "\n"
        f"Test AUC Breakdown By Class: {test_roc_auc}\n"

    )


def main():
    wandb.init(project="mimic_bert")
    df = pd.read_csv("../data/temp.csv")
    df = df.drop("Unnamed: 0", axis=1)
    if args.dev:
        df = df.head(5)
    train_df, other = train_test_split(df, test_size=0.7)
    valid_df, test_df = train_test_split(other, test_size=0.3333)

    train_dataset = MIMICDataset(train_df)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("Train dataloader created")
    valid_dataset = MIMICDataset(valid_df)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
    print("Valid dataloader created")
    test_dataset = MIMICDataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    print("Test dataloader created")

    num_labels = len(df.columns.tolist()[2:])
    net = BERTModel(num_labels)
    print("model instantiated")
    # Training!
    train(net, train_dataloader, valid_dataloader, test_dataloader, df.columns.tolist()[2:])


if __name__ == "__main__":
    main()
