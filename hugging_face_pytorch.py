import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from sklearn.metrics import accuracy_score,balanced_accuracy_score

df=pd.read_csv("train.txt",sep="\t",names=["LABEL","REVIEW"])
dataset = Dataset.from_pandas(df)
dataset=dataset.rename_columns({"LABEL":"label","REVIEW":"text"})
dataset=dataset.train_test_split(test_size=0.2,shuffle=True)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
def label_mapping(example):
    mapping={
        "DECEPTIVENEGATIVE":0,
        "DECEPTIVEPOSITIVE":1,
        "TRUTHFULNEGATIVE":2,
        "TRUTHFULPOSITIVE":3,
    }
    return {"label":mapping[example["label"]]}
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset=dataset.map(label_mapping,batched=False)
print(dataset["train"][0])
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

tokenized_datasets.set_format("torch")

from torch.utils.data import DataLoader
batch_size=8
# tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
# tokenized_datasets["test"] = tokenized_datasets["test"].shuffle(seed=42).select(range(10))
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
eval_dataloader = DataLoader(tokenized_datasets["test"], shuffle=True, batch_size=batch_size)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=4)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-5)

from transformers import get_scheduler

num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", 
    optimizer=optimizer, 
    num_warmup_steps=600, 
    num_training_steps=num_training_steps
)
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

def train_loop():
  model.train()
  for batch in train_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      progress_bar.update(1)

def test_loop():
    model.eval()
    preds=np.array()
    true_labels=np.array()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        preds=np.concatenate((preds,predictions))
        true_labels=np.concatenate((true_labels,batch["labels"]))
    print(accuracy_score(y_pred=preds,y_true=true_labels))
    print(balanced_accuracy_score(y_pred=preds,y_true=true_labels))

import evaluate
metric = evaluate.load("accuracy")
for epoch in range(1,num_epochs+1):
  print(f"Epoch {epoch}")
  train_loop()
  test_loop()
  print("Val accuracy",metric.compute()["accuracy"])




