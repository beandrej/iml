# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:

import os
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that 
# are not required 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  
NUM_EPOCHS = 15

train_embeddings_path = 'task4/train_embeddings.pt'
test_embeddings_path = 'task4/test_embeddings.pt'
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class ReviewDataset(Dataset):

    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.title_tokens = tokenizer(data_frame['title'].tolist(), padding=True, truncation=True, return_tensors="pt")
        self.sentence_tokens = tokenizer(data_frame['sentence'].tolist(), padding=True, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.data_frame)

    def concat(self, title, sentence):
        return torch.cat((title, sentence))

    def __getitem__(self, index):
        id = self.concat(self.title_tokens['input_ids'][index], self.sentence_tokens['input_ids'][index])
        mask = self.concat(self.title_tokens['attention_mask'][index], self.sentence_tokens['attention_mask'][index])

        item = {'input_ids': id, 'attention_mask': mask}

        if 'score' in self.data_frame.columns:
            score = torch.tensor(self.data_frame['score'].iloc[index], dtype=torch.float)
            item['score'] = score

        return item

def embeddings_gen():

    train_df = pd.read_csv("task4/train.csv")
    test_df = pd.read_csv("task4/test_no_score.csv")

    train_dataset = ReviewDataset(train_df)
    test_dataset = ReviewDataset(test_df)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False, pin_memory=True)

    model = RobertaModel.from_pretrained('roberta-base')
    model.to(DEVICE)
    model.eval() 

    with torch.no_grad():
        embeddings = []
        for batch in tqdm(train_loader, desc='Train Embeddings'):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:,0,:].squeeze(0)
            embeddings.append(cls_embeddings.cpu())
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, train_embeddings_path)
        print("Training Embeddings saved to train_embeddings.pt")

        embeddings = []
        for batch in tqdm(test_loader, desc='Test Embeddings'):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            cls_embeddings = outputs.last_hidden_state[:,0,:].squeeze(0)
            embeddings.append(cls_embeddings.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, test_embeddings_path)
        print("Test Embeddings saved to test_embeddings.pt")

class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, score=None):
        self.embeddings = embeddings
        self.score = score

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if not self.score == None:
            return {'embeddings': self.embeddings[idx], 'score': self.score[idx]}
        else:
            return {'embeddings': self.embeddings[idx]}

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(768, 128)
        self.a0 = nn.LeakyReLU()
        self.l1 = nn.Linear(128, 32)
        self.a1 = nn.LeakyReLU()
        self.l2 = nn.Linear(32, 8)
        self.a2 = nn.LeakyReLU()
        self.l3 = nn.Linear(8, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.05)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.batch_norm3 = nn.BatchNorm1d(8)

    def forward(self, x):
        x = self.l0(x)
        x = self.batch_norm1(x)
        x = self.a0(x)
        x = self.dropout1(x)
        x = self.l1(x)
        x = self.batch_norm2(x)
        x = self.a1(x)
        x = self.dropout2(x)
        x = self.l2(x)
        x = self.batch_norm3(x)
        x = self.a2(x)
        x = self.dropout3(x)
        x = self.l3(x)
        x = torch.clamp(x, 0, 10)
        return x

def validate_model(val_loader, model, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['embeddings'].to(DEVICE), batch['score'].to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def train_model(train_loader, val_loader, learning_rate, weight_decay):

    model = Net().to(DEVICE)

    epoch = 0
    esli = 0
    best_loss = float("inf")
    best_model = None

    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

    while True:
        model.train()
        epoch_loss = 0.0
        for _, batch in enumerate(train_loader):
            embeddings = batch['embeddings'].to(DEVICE)
            labels = batch['score'].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(embeddings)

            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        val_loss = validate_model(val_loader, model, criterion)
    
        scheduler.step()  # Update the learning rate

        if(val_loss < best_loss):
            best_loss = val_loss
            esli = 0
            best_model = model
        if(esli > 10 or epoch >= NUM_EPOCHS-1):
            print("Training Finished")
            break

        epoch += 1
        esli += 1

        print(f"Sample predictions: {outputs[:5].squeeze().detach().cpu().numpy()}")
        print(f"Sample actual scores: {labels[:5].detach().cpu().numpy()}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}, Training Loss: {epoch_loss/len(train_loader)}")
        print(f"Validation Loss: {val_loss}")
    return best_model, best_loss

if __name__ == '__main__':
    if not os.path.exists(train_embeddings_path):
        embeddings_gen()

    train_embeddings_path = 'task4/train_embeddings.pt'
    test_embeddings_path = 'task4/test_embeddings.pt'

    train_embeddings = torch.load(train_embeddings_path)
    test_embeddings = torch.load(test_embeddings_path)

    train_score = torch.tensor(pd.read_csv("task4/train.csv")['score'].values, dtype=torch.float32)
    train_embeddings_dataset = EmbeddingsDataset(train_embeddings, train_score)
    test_data = EmbeddingsDataset(test_embeddings)

    train_data, val_data = train_test_split(train_embeddings_dataset, test_size=0.3, random_state=17)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    
    weight_decay = 0.0001
    learning_rate = 0.001
    best_score = float("inf")
    best_model = None

    cur_model, cur_score = train_model(train_loader, val_loader, learning_rate, weight_decay)
    if(cur_score < best_score):
        best_score = cur_score
        best_model = cur_model
    
    best_model.eval()
    with torch.no_grad():
        results = []
        for batch in test_loader:
            embeddings = batch['embeddings'].to(DEVICE)
            outputs = best_model(embeddings)
            results.append(outputs.squeeze().cpu().numpy())
    
        with open("task4/result.txt", "w") as f:
            for list in results:
                for val in list:
                    f.write(f"{val}\n")
        print("Result file saved successfully!")
