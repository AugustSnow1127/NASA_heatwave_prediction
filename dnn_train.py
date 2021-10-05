# Read Data
tr_path = 'data\AIRS.2020.01.21.L3.RetStd_IR001.v7.0.4.0.G20310102224.hdf.nc4'
tt_path = 'data\._AIRS.2020.01.04.L3.RetStd_IR001.v7.0.4.0.G20310163447.hdf.nc4'

# Import package
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
from netCDF4 import Dataset as nDS
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Utilities
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    # plot label
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=320., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval() # set model to evaluating mode
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([280, lim], [280, lim], c='b')
    plt.xlim(280, lim)
    plt.ylim(280, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

# Dataset
class heatwaveDataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode

        # Init data
        data = torch.zeros([180, 360, 1, features_num], dtype = torch.float64)
        files = []
        
        for dirPath, dirNames, fileNames in os.walk("data", topdown = True):
            fileNames.sort()
            for f in fileNames:
                files.append(os.path.join(dirPath, f))

        # Read and preprocess daily air temperature and humidity data
        if mode == 'train' or mode == 'dev':
            begin = 1
            end = 349
        elif mode == 'test':
            begin = 349
            end = len(files)

        files = files[begin:end]

        for file in files:
            # skip DS_Store
            if file == 'yusiang_data/.DS_Store':
                continue
            else:
                nasa_data = nDS(file, mode='r')
        
            SurfSkinTemp_A = torch.tensor(nasa_data.variables['SurfSkinTemp_A'][:,:].astype(float))
            RelHumSurf_A = torch.tensor(nasa_data.variables['RelHumSurf_A'][:,:].astype(float))

            # Extend to four dimensions
            SurfSkinTemp_A = torch.unsqueeze((SurfSkinTemp_A), 2)
            RelHumSurf_A = torch.unsqueeze((RelHumSurf_A), 2)
            SurfSkinTemp_A = torch.unsqueeze((SurfSkinTemp_A), 3)
            RelHumSurf_A = torch.unsqueeze((RelHumSurf_A), 3)

            # Store all data in fifth dimension
            temp_data = torch.cat((RelHumSurf_A, SurfSkinTemp_A), 3)
            data = torch.cat((data, temp_data), 2)

        data = data[80:100, 170:190, 1:, :] # select 20 x 20 as sample
        num_of_data = data.shape[2]
        data = torch.reshape(data, (20*20*num_of_data, features_num))
        
        if mode == 'test':
            # Testing data
            self.data = data
        else:
            # Training data (train/dev sets)
            target = data[:, -1]
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
                
            self.data = data[indices]
            self.target = target[indices] 
            
            # Delete target ocean part
            for i in range(len(self.target)):
                if self.target[i] <= -9000: # ocean ~= -9999
                    self.target[i] = 0
        
        # Delete data ocean part
        for i in range(len(self.data)):
            for j in range(self.data.shape[1]):
                if self.data[i,j] <= -9000: # ocean ~= -9999
                    self.data[i,j] = 0

        # Normalize features
        self.data[:, :] = \
                    (self.data[:, :] - self.data[:, :].mean(dim=0, keepdim=True)) \
                    / self.data[:, :].std(dim=0, keepdim=True) 

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of heatwave Dataset ({} samples found, each dim = {})'.format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

# Dataloader
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = heatwaveDataset(path, mode=mode, target_only=target_only) # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True) # Construct dataloader
    return dataloader

# Deep Neural Network
class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64).float(),
            nn.ReLU().float(),
            nn.Linear(64, 1).float()
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x.float()).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L1/L2 regularization here
        return self.criterion(pred, target)

# Training
def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs from hyper-parameter

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            y = y.float()                       # cast y & pred to float
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

# Validation
def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss

# Testing
def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

# Setup Hyperparameter
device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False                   # False: Using Whole features
features_num = 2

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 200,                # maximum number of epochs
    'batch_size': 100,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.00001,               # learning rate of SGD
        'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}

# Load Data & Model
tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device

# Start Training
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

# Plotting Learning Curve
plot_learning_curve(model_loss_record, title='deep model')

del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set

# Testing & Saving Prediction
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(tt_set, model, device)  # predict heatwave cases with your model
save_pred(preds, 'pred.csv')         # save prediction file to pred.csv