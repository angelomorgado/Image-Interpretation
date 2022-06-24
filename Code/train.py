from AuxFiles.engine import train_one_epoch, evaluate
from AuxFiles.dataset import PennFudanDataset
from AuxFiles.model import get_model_instance_segmentation
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

import AuxFiles.utils as utils
import AuxFiles.transforms as T
import torch
import torchvision

def getTransform(train):
    transforms = []
    
    # converts the image, a PIL image, into a torch tensor
    transforms.append(T.ToTensor())
    
    if train:
        # During training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)


# ======================== Instantiate dataset and dataloaders ====================

# Use our dataset and defined transformations
dataset = PennFudanDataset('trainDataset', getTransform(train=True))
datasetTest = PennFudanDataset('trainDataset', getTransform(train=False))

# Split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
datasetTest = torch.utils.data.Subset(datasetTest, indices[-50:])

# Define training and validation data loaders
dataLoader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

# Define training and validation data loaders
dataLoader_test = torch.utils.data.DataLoader(
    datasetTest, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


# # ======================= Parameters ====================================
# our dataset has two classes only = background and person
numClasses = 2
numEpochs = 10

# get the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get the model using our helper function
model = get_model_instance_segmentation(numClasses)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=3, gamma=0.1)


# ====================== Train =====================================
for epoch in range(numEpochs):
    # Train for one epoch, printing every 10 iteractions
    train_one_epoch(model, optimizer=optimizer, data_loader=dataLoader, device=device, epoch=epoch, print_freq=10)
    
    # Update learning rate
    lr_scheduler.step()
    
    # Evaluate on the test dataset
    evaluate(model, data_loader=dataLoader_test, device=device)
    

torch.save(model.state_dict(), 'AuxFiles/RCNNmodel.pth')
print('Model saved with success!')