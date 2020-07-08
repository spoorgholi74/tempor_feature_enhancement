import os
import pickle
import subprocess

import torch
import numpy as np
from torch.utils.data import TensorDataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from mlp import create_model
from util_functions import adjust_lr

batch_size = 16
num_workers = 0
n_epochs = 100  # suggest training between 20-50 epochs
error_range = 0.2
transform = transforms.ToTensor()

with open('mlp_experiment/clip_time.pkl', 'rb') as fp:
    all_time = torch.Tensor(list(pickle.load(fp)))

with open('mlp_experiment/features_pickle.pkl', 'rb') as fp:
    all_features = torch.Tensor(list(pickle.load(fp)))

with open('mlp_experiment/labels_pickle.pkl', 'rb') as fp:
    all_labels = list(pickle.load(fp))

# Do the train test split
validation_split = .2
shuffle_dataset = True
random_seed= 74

# Creating data indices for training and validation splits:
assert len(all_time) == len(all_features), "Not equal length of data!"
dataset_size = len(all_time)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# Load the data into dataloader
my_dataset = TensorDataset(all_features,all_time) # create your datset
general_loader = torch.utils.data.DataLoader(my_dataset)

train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
# Create the model and train
dataiter = iter(train_loader)
images, labels = dataiter.next()

model, criterion, optimizer = create_model()
model.train()
adjustable_lr = 0.001
for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    if epoch % 30 == 0 and epoch > 0:
        adjustable_lr = adjust_lr(optimizer, adjustable_lr)
        print("New lr = %s" % adjustable_lr)

    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)

    # print training statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch + 1,
        train_loss
    ))

# Extract the features
features_list = []
for data, target in general_loader:
    output = model.embedded(data)
    output = output.view(1, -1)
    out = np.squeeze(output.cpu().detach().numpy())
    features_list.append(out)
# if os.path.exists('./mlp_experiment/time_enhanced_features.pkl') == True:
#     subprocess.call("rm -r ./mlp_experiment/time_enhanced_features.pkl")
concat_features = []
# for i in range(0, len(features_list)):
#     concat_features += features_list[i]
with open('./mlp_experiment/time_enhanced_features.pkl', 'wb') as fp:
    pickle.dump(features_list, fp)

model.eval()  # prep model for *evaluation*
test_loss = 0.0
for data, target in validation_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    correct_pred = 0
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)
    for i in range(batch_size):
        if i < len(target.data):
            pred_error = abs(target.data[i] - output.data[i])
            if pred_error <= error_range:
                correct_pred += 1
    print("Number of correct prediction from %s sample is %s" % (batch_size, correct_pred))

# calculate and print avg test loss
test_loss = test_loss / len(validation_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))