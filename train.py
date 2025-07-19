import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import glob
import os
from utils import *
from model import *
from layers import *
from sklearn.preprocessing import MinMaxScaler
import shutil
import sys
import argparse as Ap
class CombinedLossWithContribution(nn.Module):
    def __init__(self, alpha=0.5, C=1.0):
        super(CombinedLossWithContribution, self).__init__()
        self.alpha = alpha  # Balance between loss terms (MAE and MSE)
        self.C = C  # Contribution factor
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets, contribution_weights):
        mae_loss = self.mae(outputs, targets)
        mse_loss = self.mse(outputs, targets)
        combined_loss = self.alpha * mae_loss + (1 - self.alpha) * mse_loss
        return (self.C * contribution_weights * combined_loss).mean()
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, output, target):
        loss = self.alpha * self.mse(output, target) + (1 - self.alpha) * self.mae(output, target)
        return loss


seed = 0x6a09e667f3bcc908
np.random.normal(seed & 0xFFFFFFFF)
torch.manual_seed(seed & 0xFFFFFFFF)

argp = Ap.ArgumentParser()
argp.add_argument("--tct", default='chicago', type=str, help="Target city")
argp.add_argument("--tr", default=7, type=int, help="Target region")
argp.add_argument("--tc", default=1, type=int, help="Target category")
argp.add_argument("--bs", default=42, type=int, help="Batch size")
argp.add_argument("--ts", default=120, type=int, help="Number of time steps")
argp.add_argument("--rts", default=20, type=int, help="Number of recent time steps")
argp.add_argument("--ncf", default=1, type=int, help="Number of crime features per time step")
argp.add_argument("--nxf", default=12, type=int, help="Number of external features per time step")
argp.add_argument("--gout", default=8, type=int, help="Dimension of output features of GAT")
argp.add_argument("--gatt", default=40, type=int, help="Dimension of attention module of GAT")
argp.add_argument("--rhid", default=40, type=int, help="Dimension of hidden state of SAB-LSTMs")
argp.add_argument("--ratt", default=30, type=int, help="Dimension of attention module of SAB-LSTMs")
argp.add_argument("--rl", default=1, type=int, help="Number of layers of SAB-LSTMs")
d = argp.parse_args(sys.argv[1:])

target_city = d.tct
target_region = d.tr
target_cat = d.tc
time_step = d.ts
recent_time_step = d.rts
batch_size = d.bs
gat_out = d.gout
gat_att = d.gatt
ncfeature = d.ncf
nxfeature = d.nxf
slstm_nhid = d.rhid
slstm_nlayer = d.rl
slstm_att = d.ratt

gen_gat_adj_file(target_city, target_region)   # generate the adj_matrix file for GAT layers
loaded_data = torch.from_numpy(np.loadtxt("data/" + target_city + "/com_crime/r_" + str(target_region) + ".txt", dtype=int)).T
loaded_data = loaded_data[:, target_cat:target_cat+1]
x, y, x_daily, x_weekly = create_inout_sequences(loaded_data)

scale = MinMaxScaler(feature_range=(-1, 1))
x = torch.from_numpy(scale.fit_transform(x))
x_daily = torch.from_numpy(scale.fit_transform(x_daily))
x_weekly = torch.from_numpy(scale.fit_transform(x_weekly))
y = torch.from_numpy(scale.fit_transform(y))


# Divide your data into train set & test set
train_x_size = int(x.shape[0] * .67)
train_x = x[: train_x_size, :]  # (ns_tr=num of train samples, ts)
train_x_daily = x_daily[: train_x_size, :]
train_x_weekly = x_weekly[: train_x_size, :]
train_y = y[: train_x_size, :]  # (ns_tr, 1)

test_x = x[train_x_size:, :]  # (ns_te = num of test samples, ts) = (683, ts)
test_x_daily = x_daily[train_x_size:, :]
test_x_weekly = x_weekly[train_x_size:, :]
test_x = test_x[:test_x.shape[0] - 11, :]  # 11 is subtracted to make ns_te compatible with bs
test_x_daily = test_x_daily[:test_x_daily.shape[0] - 11, :]
test_x_weekly = test_x_weekly[:test_x_weekly.shape[0] - 11, :]
test_y = y[train_x_size:, :]
test_y = test_y[:test_y.shape[0] - 11, :]

# Divide it into batches
train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step)  # (nb=num of batches, bs, rts)
train_x_daily = train_x_daily.view(int(train_x_daily.shape[0] / batch_size), batch_size, train_x_daily.shape[1])  # (nb, bs, dts)
train_x_weekly = train_x_weekly.view(int(train_x_weekly.shape[0] / batch_size), batch_size, train_x_weekly.shape[1])  # (nb, bs, rws)
train_y = train_y.view(int(train_y.shape[0] / batch_size), batch_size, 1)

test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step)
test_x_daily = test_x_daily.view(int(test_x_daily.shape[0] / batch_size), batch_size, test_x_daily.shape[1])
test_x_weekly = test_x_weekly.view(int(test_x_weekly.shape[0] / batch_size), batch_size, test_x_weekly.shape[1])
test_y = test_y.view(int(test_y.shape[0] / batch_size), batch_size, 1)

# load data for external_features and side_features
train_feat, test_feat = load_data_regions(batch_size, target_cat, target_region, target_city)
train_feat_ext, test_feat_ext = load_data_regions_external(batch_size, nxfeature, target_region, target_city)
train_crime_side, test_crime_side = load_data_sides_crime(batch_size, target_cat, target_region, target_city)

# Model and optimizer
model = AIST(ncfeature, nxfeature, gat_out, gat_att, slstm_nhid, slstm_att, slstm_nlayer, batch_size,
             recent_time_step, target_city, target_region, target_cat)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)

lr = 0.001
weight_decay = 5e-4
# When initializing your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Add weight_decay
# Initialize your custom loss function
alpha = 0.7 # Adjust this to balance between MAE and MSE
C = 1.0  # Set your initial contribution factor (can be dynamic)
high_crime_threshold = -1.0  # Example threshold; adjust as necessary
C_high = 2.0  # Weight for high crime
C_low = 0.1   # Weight for low crime
#criterion = CombinedLossWithContribution(alpha=alpha, C=C)
#criterion = CustomLoss(alpha=alpha)
criterion = nn.L1Loss()
epochs = 300
best = epochs + 1
best_epoch = 0
t_total = time.time()
loss_values = []
bad_counter = 0
patience = 100
print("the shape of train_x :",train_x.shape)
train_batch = train_x.shape[0]
test_batch = test_x.shape[0]

for epoch in range(epochs):
    i = 0
    loss_values_batch = []
    for i in range(train_batch):
        t = time.time()

        x_crime = Variable(train_x[i]).float()
        x_crime_daily = Variable(train_x_daily[i]).float()
        x_crime_weekly = Variable(train_x_weekly[i]).float()
        y = Variable(train_y[i]).float()

        model.train()
        optimizer.zero_grad()
        output, attn = model(x_crime, x_crime_daily, x_crime_weekly, train_feat[i], train_feat_ext[i], train_crime_side[i])
        y = y.view(-1, 1)

        #loss_train = criterion(output, y)
        # Define contribution_weights based on conditions
        #contribution_weights = torch.where(y > high_crime_threshold, C_high, C_low)
        
        # Calculate the custom loss
        loss_train = criterion(output, y)
        
        loss_train.backward()
        optimizer.step()

        print('Epoch: {:04d}'.format(epoch*train_batch + i + 1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'time: {:.4f}s'.format(time.time() - t))

        loss_values.append(loss_train)
        torch.save(model.state_dict(), '{}.pkl'.format(epoch*train_batch + i + 1))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch*train_batch + i + 1
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    if epoch*train_batch + i + 1 >= 800:
        break
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

#best_epoch = -1
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
f = open('result/aist.txt','a')


stat_y = []
stat_y_prime = []

def compute_test():
    loss = 0
    all_y_true = []  # Initialize list to store true values
    all_y_pred = [] 
    for i in range(test_batch):
        model.eval()

        # Prepare the test data
        x_crime_test = Variable(test_x[i]).float()
        x_crime_daily_test = Variable(test_x_daily[i]).float()
        x_crime_weekly_test = Variable(test_x_weekly[i]).float()
        y_test = Variable(test_y[i]).float()

        # Get model predictions
        output_test, list_att = model(x_crime_test, x_crime_daily_test, x_crime_weekly_test, test_feat[i], test_feat_ext[i], test_crime_side[i])
        y_test = y_test.view(-1, 1)
        y_test = torch.from_numpy(scale.inverse_transform(y_test))
        output_test = torch.from_numpy(scale.inverse_transform(output_test.detach()))

        stat_y.append(y_test.detach().numpy())
        stat_y_prime.append(output_test.numpy())
        # Append values to all_y_true and all_y_pred
        all_y_true.extend(y_test.numpy().flatten())
        all_y_pred.extend(output_test.numpy().flatten())

        # Define contribution weights for test based on crime levels
        #contribution_weights = torch.where(y_test > high_crime_threshold, C_high, C_low)

        # Calculate loss for the test set
        loss_test = criterion(output_test, y_test)

        # Print results for each batch
        loss += loss_test.data.item()
        print(f"Test set results: loss= {loss_test.data.item():.4f}")
    

    
    print(f"{target_region} {target_cat} {loss / i:.4f}")
    print(f"{target_region} {target_cat} {loss / i:.4f}", file=f)
    # Calculate regression metrics
    mae = mean_absolute_error(all_y_true, all_y_pred)
    mse = mean_squared_error(all_y_true, all_y_pred)
    r2 = r2_score(all_y_true, all_y_pred)
    mape = mean_absolute_percentage_error(all_y_true, all_y_pred)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

# Run the test evaluation
compute_test()

