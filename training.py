import numpy as np
import torch
import gc
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.autograd import Variable
import dataset
import net
import easy_net
import params
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.utils import class_weight

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_val_dataset(d, val_split=0.25):
    train_idx, val_idx = train_test_split(list(d.data.index), test_size=val_split, stratify=d.labels)
    return Subset(d, train_idx), Subset(d, val_idx)


print(f"Preparing...")
ds = dataset.AmazonDataset()  # fare prova con anche dataset non caricato
ds.load_dataset()
ds.filter()
weights = class_weight.compute_class_weight('balanced', classes=np.unique(ds.labels), y=ds.labels)
weights = torch.Tensor(weights)
train_ds, test_ds = train_val_dataset(ds)

# Define parameters
batch_size = params.BATCH_SIZE
hidden_dim = 256
embedding_size = params.NUM_FEATURES
dropout_rate = params.DROPOUT_RATE
lr = params.LR
epochs = params.NUM_EPOCHS
best_loss = float('inf')


train_loader = DataLoader(train_ds, batch_size=params.BATCH_SIZE, collate_fn=dataset.collate_batch,
                          shuffle=True, num_workers=0, drop_last=True)
# Build the model
lstm_model = net.SentimentAnalysis(batch_size,
                                   hidden_dim,
                                   embedding_size,
                                   dropout_rate)  # modificato LSTM con SentimentAnalysis (nome rete)

lstm_model.to(device)
# optimization algorithm
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

to_train = True
ce = nn.CrossEntropyLoss(weight=weights)
bce = nn.BCEWithLogitsLoss(pos_weight=weights)
mse = nn.MSELoss()
softmax = torch.nn.Softmax(dim=1)

lstm_model.train()

# train and validate
print("Start running")
if to_train:
    for epoch in range(epochs):
        # training
        epoch_loss = 0; epoch_acc = 0; epoch_f1 = 0; epoch_precision = 0; epoch_recall = 0
        print(f"Epoch: {epoch}")
        for idxs, (batch, labels) in enumerate(train_loader):
            #print(f"epoch = {epoch}, idxs = {idxs}")
            gc.collect()
            optimizer.zero_grad()
            batch = batch.to(device)
            labels = labels.to(device)
            predictions = lstm_model(batch)
            labels = torch.Tensor([x - 1 for x in labels.data.numpy()])  # mapping classes 1-5 in 0-4
            long_labels = labels.type(torch.LongTensor)
            loss1 = 0.5 * ce(predictions, long_labels)
            # loss1 = 0.5 * bce(predictions, nn.functional.one_hot(long_labels).float())
            float_preds = torch.argmax(softmax(predictions), 1)
            float_preds = float_preds.type(torch.FloatTensor)
            loss2 = torch.sqrt(mse(float_preds, labels))
            loss1 = Variable(loss1, requires_grad=True)
            loss2 = Variable(loss2, requires_grad=True)
            loss = loss1 + loss2
            print(f"Loss: {loss}")
            #print(f"float_preds = {float_preds}")
            #print(f"labels = {labels}")
            accuracy = accuracy_score(float_preds.detach().data, long_labels)
            f1 = f1_score(float_preds.detach().data, long_labels, average='weighted', labels=np.unique(long_labels))
            precision = precision_score(float_preds.detach().data, long_labels, average='weighted', labels=np.unique(long_labels))
            recall = recall_score(float_preds.detach().data, long_labels, average='weighted', labels=np.unique(long_labels))
            loss1.backward()
            loss2.backward()
            optimizer.step()
            # loss = loss1.detach().item() + loss2.detach().item()
            #loss = loss1.detach().item() + loss2.detach().item()
            #print(f"Loss: {loss}")
            epoch_loss = epoch_loss + loss
            epoch_acc = epoch_acc + accuracy
            epoch_f1 = epoch_f1 + f1
            epoch_precision = epoch_precision + precision
            epoch_recall = epoch_recall + recall

        train_loss, train_acc = epoch_loss / len(train_loader), epoch_acc / len(train_loader)
        train_f1, train_precision = epoch_f1 / len(train_loader), epoch_precision / len(train_loader)
        train_recall = epoch_recall / len(train_loader)
        # save best model
        if train_loss < best_loss:
            best_valid_loss = train_loss
            torch.save(lstm_model.state_dict(), 'saved_weights_BiLSTM.pt')
        print(f'\tEpoch; {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f"F1 = {train_f1}, precision = {train_precision}, recall = {train_recall}")

# load weights and make predictions
lstm_model.load_state_dict(torch.load("saved_weights_BiLSTM.pt"))
epoch_loss = 0
epoch_acc = 0

lstm_model.eval()

test_loader = DataLoader(test_ds, batch_size=params.BATCH_SIZE, collate_fn=dataset.collate_batch,
                         shuffle=True, num_workers=0, drop_last=True)

with torch.no_grad():
    for idxs, (batch, labels) in enumerate(test_loader):
        predictions = lstm_model(batch)
        labels = torch.Tensor([x - 1 for x in labels.data.numpy()])  # mapping classes 1-5 in 0-4
        long_labels = labels.type(torch.LongTensor)
        loss1 = 0.5 * ce(predictions, long_labels)
        float_preds = torch.argmax(softmax(predictions), 1)
        float_preds = float_preds.type(torch.FloatTensor)
        loss2 = 0.5 * mse(float_preds, labels)
        loss = loss1.clone() + loss2.clone()
        loss = Variable(loss, requires_grad=True)
        accuracy = accuracy_score(float_preds.data, long_labels)
        epoch_loss += loss
        epoch_acc += accuracy

test_loss, test_acc = epoch_loss / len(test_loader), epoch_acc / len(test_loader)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
