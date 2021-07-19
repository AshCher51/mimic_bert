import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class MIMICDataset(torch.data.utils.Dataset):
    def __init__(self, texts, labels, max_len=128):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        tokenized_text = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_len,
        )
        ids = tokenized_text["input_ids"]
        mask = tokenized_text["attention_mask"]
        token_type_ids = tokenized_text["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.tensor(label, dtype=torch.float),
        }


class BERTModel(nn.Module):
    def __init__(self, num_labels):
        super(BERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", return_dict=False
        )
        self.dropout = nn.Dropout(0.1)
        self.hidden = nn.Linear(768, 64)
        self.classifier = nn.Linear(64, num_labels)

    def forward(self, model_dict):
        _, pooled_output = self.bert(model_dict['input_ids'], model_dict['token_type_ids'], model_dict['attention_mask'])
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train(model, train_dataloader, valid_dataloader, test_dataloader, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            target = batch['target'].to(device)

            logits = model(ids, mask, token_type_ids)
            loss = criterion(logits, batch['target'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * train_dataloader.size(0)
        train_loss /= len(train_dataloader.sampler)

        # Validation
        model.eval()
        valid_loss = 0
        for batch in valid_dataloader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            target = batch['target'].to(device)

            logits = model(ids, mask, token_type_ids)
            loss = criterion(logits, batch['target'])
            valid_loss += loss.item() * train_dataloader.size(0)
        valid_loss = valid_loss / len(valid_dataloader.sampler)

        print(
            f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {train_loss:.3f}.. "
            f"Validation loss: {valid_loss:.3f}"
        )

def main():
    df = pd.read_csv('../data/temp.csv')
    train_df, other = train_test_split(df, test_size=0.7)
    valid_df, test_df = train_test_split(other, test_size=0.3333)
    
    train_dataset = MIMICDataset(train_df.TEXT.values, train_df.target.values)




if __name__ == '__main__':
    main()