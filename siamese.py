# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnext101_32x8d, ResNeXt101_32X8D_Weights
import torch.optim as optim

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")
# When using the GPU, it is important that your model and all data are on the 
# same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
embeddings_path = 'dataset/embeddings.npy'

def generate_embeddings():

    train_transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()
    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=False,
                              pin_memory=True, num_workers=16)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.to(device)
    embedding_size = 2048 
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))

    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    with torch.no_grad():
        index = 0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            embeddings_batch = model(X_batch).cpu().numpy().squeeze()
            batch_size = len(X_batch)
            embeddings[index:index+batch_size, :] = embeddings_batch
            index += batch_size

    np.save(embeddings_path, embeddings)

def get_data(file, train=True, cutoff=[-1000000, 10000000]):

    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load(embeddings_path)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        ts = t.split()
        if(int(ts[0]) < cutoff[0] or int(ts[0]) >= cutoff[1]): continue
        if(int(ts[1]) < cutoff[0] or int(ts[1]) >= cutoff[1]): continue
        if(int(ts[2]) < cutoff[0] or int(ts[2]) >= cutoff[1]): continue
        emb = [file_to_embedding[a] for a in ts]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y


def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)  # L2 distance
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()

def validate_model(model, val_loader):
    model.eval()
    total_misclassifications = 0
    total_comparisons = 0

    with torch.no_grad():
        for [X, y] in val_loader:
            anchor, positive, negative = X[:,:2048], X[:, 2048:4096], X[:, 4096:6144]
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Get the model's embeddings for the anchor, positive, and negative samples
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            # Compute squared L2 distances for positive and negative pairs
            dist_pos = (anchor_embed - positive_embed).pow(2).sum(1)
            dist_neg = (anchor_embed - negative_embed).pow(2).sum(1)

            # Increment misclassifications where the positive distance is not less than the negative distance
            misclassifications = (dist_pos >= dist_neg).sum().item()
            total_misclassifications += misclassifications
            total_comparisons += anchor.size(0)

    # Calculate the misclassification rate
    misclassification_rate = total_misclassifications / total_comparisons if total_comparisons > 0 else 0
    return misclassification_rate

def train_model(train_loader, val_loader, test_loader, seed, weight_decay, learning_rate):

    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    
    torch.manual_seed(seed)
    model = SiameseNet()
    model.train()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    esli = 0
    best_loss = float("inf")
    best_model = None
    epoch = 0
    while(True):  
        model.train()
        epoch_loss = .0
        for [X, y] in train_loader:
            anchor, positive, negative = X[:,:2048], X[:, 2048:4096], X[:, 4096:6144]
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            
            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Training Epoch: {epoch+1} Loss: {epoch_loss}')  
        
        #validate
        val_loss = validate_model(model, val_loader)
        print(f"Validation Loss: {1.0-val_loss}")
        #early stopping if loss doesnt improve in the last 10 epochs
        esli += 1
        if(val_loss < best_loss):
            best_loss = val_loss
            esli = 0
            best_model = model
        if(esli > 10 or epoch >= 11):
            print("Stopping early due to lack of improvement")
            break
        epoch += 1

    test_model(model, test_loader, f"results/Score_{1.0-best_loss}_wd_{weight_decay}_lr_{learning_rate}_seed_{seed}.txt")
    return best_model, best_loss

def test_model(model, loader, filename = "results.txt"):

    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [X] in loader:

            anchor, positive, negative = X[:,:2048], X[:, 2048:4096], X[:, 4096:6144]
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Get the model's embeddings for the anchor, positive, and negative samples
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            # Compute squared L2 distances for positive and negative pairs
            dist_pos = (anchor_embed - positive_embed).pow(2).sum(1)
            dist_neg = (anchor_embed - negative_embed).pow(2).sum(1)
            prediction = (dist_pos < dist_neg).int()
            predictions.append(prediction.cpu().numpy())

    predictions = np.concatenate(predictions)

    print(predictions)
    np.savetxt(filename, predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists(embeddings_path) == False):
        print("Generating Embeddings")
        generate_embeddings()

    print("Getting Training Data")
    X, y = get_data(TRAIN_TRIPLETS)
    X_train, y_train = X[0:50000], y[0:50000]
    X_val, y_val = X[50000:] , y[50000:]
    print(f"Training Set Size: {len(X_train)}")
    print(f"Validation Set Size: {len(X_val)}")

    print("Creating Training Loader")
    dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                            torch.from_numpy(y).type(torch.long))
    train_loader = create_loader_from_np(X, y, train=True, batch_size=32, shuffle=True, num_workers=4)
    val_loader = create_loader_from_np(X_val, y_val, train=True, batch_size=32, shuffle=False, num_workers=4)
    # delete the loaded training data to save memory, as the data loader copies
    del X
    del y

    # repeat for testing data
    print("Getting Test Data")
    X_test, y_test = get_data(TEST_TRIPLETS, train=False)
    print("Creating Test Loader")
    test_loader = create_loader_from_np(X_test, train = False, batch_size=1000, shuffle=False)
    del X_test
    del y_test

    # define a model and train it

    seeds_matrix = [100]
    weight_decay_matrix = [0]
    learning_rate_matrix = [0.00001]
    best_score = float("inf")
    best_model = None
    for seed in seeds_matrix:
        for weight_decay in weight_decay_matrix:
            for learning_rate in learning_rate_matrix:
                print(f"Training Model:\nSeed: {seed}\nWeight Decay: {weight_decay}\nLearning Rate: {learning_rate}")
                cur_model, cur_score = train_model(train_loader, val_loader, test_loader, seed, weight_decay, learning_rate)
                if(cur_score < best_score):
                    best_score = cur_score
                    best_model = cur_model
                    print("New Best Model")

    
    # test the model on the test data
    print("Testing Model")
    test_model(best_model, test_loader)
    print("Results saved to results.txt")
