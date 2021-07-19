import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from transformers import (
    BertTokenizerFast as BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)


class MIMICDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 128):
        super().__init__()

        self.max_token_len = max_token_len
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data_row = self.data.iloc[idx]

        report_text = data_row.TEXT
        labels = data_row[LABEL_COLUMNS]

        encoding = self.tokenizer.encode_plus(
            report_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            report_text=report_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )

class MIMICModel(nn.Module):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELossWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        

def run():
    data = pd.read_csv("../data/temp.csv")
    train_df, other = train_test_split(data, test_size=0.7)
    valid_df, test_df = train_test_split(other, test_size=0.667)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



if __name__ == "__main__":
    run()
