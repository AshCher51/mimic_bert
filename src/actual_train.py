import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

print("Packages loaded")

# Parse the Args
parser = argparse.ArgumentParser(description='Finetune Bert Model for Downstream Text Classification task.')
parser.add_argument('--dev', type=bool, 
    help="Whether or not to train on whole dataset"
)
parser.add_argument('--epochs', type=int,
    help="Number of epochs to train model"
)
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
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        ids = tokenized_text["input_ids"].flatten().numpy()
        mask = tokenized_text["attention_mask"].flatten().numpy()
        # token_type_ids = tokenized_text["token_type_ids"].flatten().numpy()

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
        #    "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.FloatTensor(labels),
        }


class BERTModel(nn.Module):
    def __init__(self, num_labels):
        super(BERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            return_dict=True
        )
        self.dropout = nn.Dropout(0.1)
        if not args.dev:
            self.hidden = nn.Linear(768, 64)
            self.classifier = nn.Linear(64, num_labels)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, model_dict):
        out = self.bert(model_dict['input_ids'], model_dict['attention_mask'])#, model_dict['token_type_ids'])
        # print(out)
        pooled_output = self.dropout(out.pooler_output)
        if not args.dev:
            pooled_output = self.hidden(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train(model, train_dataloader, valid_dataloader, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    print("start of training loop")
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        count = 0
        for batch in train_dataloader:
            optimizer.zero_grad()

            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            # token_type_ids = batch['token_type_ids'].to(device)
            target = batch['target'].to(device)
            input_dict = {"input_ids":ids, "attention_mask":mask}
            logits = model(input_dict)# token_type_ids)
            loss = criterion(logits, batch['target'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch['ids'].size(0)

            # Metrics - TODO add various metrics
            probs = torch.sigmoid(logits).cpu().detach()
            predicted = (probs>0.5).float().numpy()
            actual = batch['target'].cpu().numpy()

            train_acc += accuracy_score(actual, predicted)
            count += 1
        train_loss /= len(train_dataloader.sampler)
        train_acc /= count

        # Validation
        model.eval()
        valid_loss = 0
        valid_acc = 0
        count = 0
        for batch in valid_dataloader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            # token_type_ids = batch['token_type_ids'].to(device)
            target = batch['target'].to(device)

            input_dict = {"input_ids":ids, "attention_mask":mask}
            logits = model(input_dict) # token_type_ids)
            loss = criterion(logits, batch['target'])
            valid_loss += loss.item() * batch['ids'].size(0)

            # Metrics - TODO add various metrics
            probs = torch.sigmoid(logits).cpu().detach()
            predicted = (probs>0.5).float().numpy()
            actual = batch['target'].cpu().numpy()

            valid_acc += accuracy_score(actual, predicted)
            count += 1

        valid_loss /= len(valid_dataloader.sampler)
        valid_acc /= count

        print(
            "\n"*3
            f"Epoch {epoch+1}/{args.epochs}.. "
            f"Train loss: {train_loss:.3f}.. "
            f"Validation loss: {valid_loss:.3f}.."
            "\n"
            f"Train accuracy: {train_acc:.3f}"
            f"Validation accuracy: {valid_acc:.3f}"
        )

    # Testing
    model.eval()
    test_loss = 0
    test_acc = 0
    count = 0
    for batch in test_dataloader:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        # token_type_ids = batch['token_type_ids'].to(device)
        target = batch['target'].to(device)

        input_dict = {"input_ids":ids, "attention_mask":mask}
        logits = model(input_dict)# token_type_ids)
        loss = criterion(logits, batch['target'])
        test_loss += loss.item() * batch['ids'].size(0)

        # Metrics - TODO add various metrics
        probs = torch.sigmoid(logits).cpu().detach()
        predicted = (probs>0.5).float().numpy()
        actual = batch['target'].cpu().numpy()

        test_acc += accuracy_score(actual, predicted)
        count += 1

    valid_loss /= len(valid_dataloader.sampler)
    valid_acc /= count
    print(
            "Training Finished!\n"
            "Final Metrics: \n"
            f"Train loss: {train_loss:.3f} \n"
            f"Validation loss: {valid_loss:.3f}\n"
            "\n"
            f"Train accuracy: {train_acc:.3f}\n"
            f"Validation accuracy: {valid_acc:.3f}\n"
            f"Test accuracy: {test_acc:.3f}"
    )
def main():
    df = pd.read_csv('../data/temp.csv')
    df = df.drop('Unnamed: 0', axis=1)
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
    train(net, train_dataloader, valid_dataloader, test_dataloader)

if __name__ == '__main__':
    main()