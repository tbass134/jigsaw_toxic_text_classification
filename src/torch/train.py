
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator
from sklearn.metrics import accuracy_score
import tqdm
import os
from model.model import LSTM
from utils.utils import BatchWrapper



def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def split_data(size=0.8, min_rows=0, path="data/train_val_split"):
    if min_rows > 0:
        df = pd.read_csv("data/train.csv", nrows=min_rows)
    else:
        df = pd.read_csv("data/train.csv")

    mask = np.random.rand(len(df)) < size
    if not os.path.exists(path):
        os.mkdir(path)

    df[mask].to_csv(os.path.join(path,"train.csv"), index=False)
    df[~mask].to_csv(os.path.join(path,"val.csv"), index=False)


def train(train_dl):
    loss = 0
    for x, y in train_dl:
        opt.zero_grad()
        preds = model(x)
        loss = loss_func(y, preds)
        loss.backward()
        loss.step()
        loss +=loss.item()
    return loss

def val(val_dl):
    loss = 0
    for x, y in val_dl:
        preds = model(x)
        loss = loss_func(y, preds)
        loss +=loss.item()
    return loss

def run():

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    num_epochs = 5
    batch_size = 64
    data_path = "data/train_val_split"
    
    split_data(path=data_path)
  
    TEXT = Field(tokenize = 'basic_english', lower=True, sequential=True)
    LABEL = Field(sequential=False, use_vocab=False)

    fields = [
        ('id',None),  #ignore id col
        ('comment_text', TEXT), 
        ('toxic', LABEL),
        ('severe_toxic', LABEL),
        ('obscene', LABEL),
        ('threat', LABEL),
        ('insult', LABEL),
        ('identity_hate', LABEL)
    ]

    train_ds, val_ds = TabularDataset.splits(path=data_path,train="train.csv", validation="val.csv",format="csv", skip_header=True, fields=fields)

    TEXT.build_vocab(train_ds)
    LABEL.build_vocab(train_ds)

    # create iterators for train/valid/test datasets
    train_iter, val_iter = BucketIterator.splits(
        (train_ds, val_ds), # we pass in the datasets we want the iterator to draw data from
        batch_sizes=(batch_size, batch_size),
        device=device, # if you want to use the GPU, specify the GPU number here
        sort_key=lambda x: len(x.comment_text), # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=True,
        repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
    )

    train_dl = BatchWrapper(train_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    valid_dl = BatchWrapper(val_iter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

    loss_func = nn.BCEWithLogitsLoss()
    loss_func = loss_func.to(device)

    model = LSTM(vocab = TEXT.vocab, hidden_dim=500, embedding_dim=100)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-2)


    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train() # turn on training mode
        for x, y in tqdm.tqdm(train_dl): # thanks to our wrapper, we can intuitively iterate over our data!
            opt.zero_grad()

            outputs = model(x)
            loss = loss_func(outputs, y)
            loss.backward()
            opt.step()

            train_loss += loss.item()

        # calculate the validation loss for this epoch
        val_loss = 0.0
        model.eval() # turn on evaluation mode
        for x, y in tqdm.tqdm(valid_dl): # thanks to our wrapper, we can intuitively iterate over our data!
            outputs = model(x)
            loss = loss_func(outputs, y)

            val_loss += loss.item()

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, train_loss, val_loss))
        torch.save(model,"model.bin")

if __name__ == "__main__":
    run()