import pickle
import random
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch import nn, optim
from tqdm import tqdm

class GRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout):
        super(GRU, self).__init__()
        self.cell = nn.GRU(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x, _ = self.cell(inputs)
        outputs = x.clone()
        h_n = _[0]
        x = self.linear(x[:, -1, :])
        return x, outputs, h_n

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(df, batch_size, generator, train_size, validation_size, test_size):
    df = df['Close'].values
    df = df.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_df = scaler.fit_transform(df)

    x_train = []
    y_train = []
    for i in range(60,len(scaled_df)):
        x_train.append(scaled_df[i-60:i,0])
        y_train.append(scaled_df[i,0])
    x_train = np.array(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = np.array(y_train)

    dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, validation_size, test_size], generator=generator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=generator)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=generator)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=generator)
    return (train_dataloader, validation_dataloader, test_dataloader)

def train(model, train_loader, validation_loader, n_epochs, optimizer):
    loss_fn = nn.MSELoss()
    training_losses, validation_losses = [], []
    for _ in tqdm(range(n_epochs)):
        epoch_loss = 0
        model.train()

        for x, y in train_loader:
            preds, _, _ = model(x)  # forward pass
            loss = loss_fn(preds[:, 0], y)  # computing the loss
            optimizer.zero_grad()  # zeroing the gradients of the model parameters
            loss.backward()  # backward pass
            optimizer.step()  # model parameters

            epoch_loss += loss.item()
        print('\t Epoch Train Loss: ', epoch_loss / len(train_loader))
        training_losses.append(epoch_loss/len(train_loader))

        epoch_loss = 0
        model.eval()
        for x, y in validation_loader:
            preds, _, _ = model(x)  # forward pass
            loss = loss_fn(preds[:, 0], y)  # computing the loss

            epoch_loss += loss.item()
        print('\t Epoch valid Loss: ', epoch_loss / len(validation_loader))
        validation_losses.append(epoch_loss/len(validation_loader))

    return model, training_losses, validation_losses


def test(model, test_dataloader):
    # We use BinaryCrossEntropyLoss for our logistic regression task
    loss_fn = nn.MSELoss()

    X, Y = next(iter(test_dataloader))[0], next(iter(test_dataloader))[1]
    # Our predictions
    outputs = model(X)[0]

    predictions = torch.round(torch.sigmoid(outputs))

    # Losses
    loss = loss_fn(outputs[:, 0], Y)  # computing the loss

    # We need numpy arrays for metrics
    predictions, labels = predictions.cpu().detach().numpy(), Y.cpu().detach().numpy()

    return (metrics.mean_squared_error(predictions, labels),
            metrics.mean_absolute_error(predictions, labels),
            metrics.r2_score(predictions, labels),
            loss/len(test_dataloader))


def train_model(seed=0,
                batch_size=64,
                learning_rate=1e-3,
                train_size=8000,
                validation_size=500,
                test_size=1500,
                n_epochs=10,
                **kwargs):

    # We ensure our results are reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # We get our dataset
    df = pd.read_csv('data/AMZN.csv')
    print(len(df))
    data = get_dataloaders(df, batch_size, generator,
                           train_size, validation_size, test_size)

    # We construct our model
    model = GRU(input_size=1, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # We train and test our model
    model, train_losses, validation_losses = train(model, data[0], data[1], n_epochs, optimizer)
    mse, mae, r2, test_loss = test(model, data[2])

    # We save our model
    pickle.dump(model, open("models/gru_model", "wb"))

    return model, (train_losses, validation_losses), (mse, mae, r2), test_loss


if __name__ == '__main__':
    # train
    model_parameters = {
        'hidden_size': 200,
        'num_layers': 3,
        'dropout': 0.2
    }
    model, losses, metrics, test_loss = train_model(
        seed=0,
        batch_size=64,
        train_size=5135,
        validation_size=500,
        test_size=500,
        n_epochs=3,
        **model_parameters)
    mse, mae, r2 = metrics
    print("MSE: %f" % (mse))
    print("MAE: %f" % (mae))
    print("R2 score: %f" % (r2))