# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import matplotlib
import numpy as np
import datetime as dt
import torchvision
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50, ResNet18_Weights, resnet18
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")
# When using the GPU, it is important that your model and all data are on the 
# same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    train_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory
    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size= batch_size,
                              shuffle=False,
                              pin_memory=True, num_workers=8)
    it = iter(train_loader)
    images, labels = next(it)
    imshow(torchvision.utils.make_grid(images))

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)
    embeddings = []
    embedding_size = 512 # Dummy variable, replace with the actual embedding size once you
    # pick your model
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))
    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates.
    # Initialize model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    print(model)
    layer = model._modules.get('avgpool')
    outputs = []
    def copy_embeddings(m, i, o):
        """Copy embeddings from the penultimate layer.
        """
        print(o.shape)
        ls = o[:, :, 0, 0].detach().numpy().tolist()
        for i in range(o.shape[0]):
            outputs.append(o[i,:,0,0].detach().numpy().tolist())


    _ = layer.register_forward_hook(copy_embeddings)

    # Set model to eval mode
    model.eval()

    with torch.no_grad():
        for idx, (data, target) in enumerate(train_loader):
            print('Batch index: ', idx)
            model(data)
    np.save('dataset/embeddings.npy', outputs)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('\\')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings.npy')
    # TODO: Normalize the embeddings across the dataset
    print(embeddings[0])
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    embeddings = (embeddings - mean) / std
    #rowsums = embeddings.sum(axis=1)
    #embeddings = embeddings / rowsums[:, np.newaxis]
    print(embeddings[0])
    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you 
# don't run out of memory (VRAM if on GPU, RAM if on CPU)
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        # Attention: If you get type errors you can modify the type of the
        # labels here
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader


# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """

    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(6144, 1000, bias=True)
        self.fc2 = nn.Linear(1000, 500, bias=True)
        self.fc3 = nn.Linear(500, 100, bias=True)
        self.fc4 = nn.Linear(100, 1, bias=True)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x
    
def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    # Define the model
    model = Net()
    model.train()  # Set the model to training mode
    model.to(device)  # Move the model to the appropriate device (CPU or GPU)
    
    # Define hyperparameters
    learning_rate = 0.001
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
    
    num_epochs = 10  # Number of training epochs
    
    for epoch in range(num_epochs):
        # Iterate over the batches in the training data loader
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move input and target to the appropriate device
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x_batch)
            
            # Calculate loss
            loss = criterion(outputs, y_batch.unsqueeze(1).float())  # BCEWithLogitsLoss expects target to be 2D
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
        # Print epoch loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Return the trained model
    return model


def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training data
    X, y = get_data(TRAIN_TRIPLETS)
    # Create data loaders for the training data
    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)
    # delete the loaded training data to save memory, as the data loader copies
    del X
    del y

    # repeat for testing data
    X_test, y_test = get_data(TEST_TRIPLETS, train=False)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)
    del X_test
    del y_test

    # define a model and train it
    model = train_model(train_loader)
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
