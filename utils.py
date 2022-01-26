<<<<<<< HEAD
import torch
from torch.autograd import Variable
from torchtext import data
import matplotlib.pyplot as plt

def accuracy(probs, target):
  winners = probs.argmax(dim=1)
  corrects = (winners == target)
  accuracy = corrects.sum().float() / float(target.size(0))
  return accuracy

######################################## Using torchText ######################################

def create_iterator(train_data, valid_data, test_data, batch_size, device):
    #  BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    # by setting sort_within_batch = True.
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
        batch_size = batch_size,
        sort_key = lambda x: len(x.text), # Sort the batches by text length size
        sort_within_batch = True,
        device = device)
    return train_iterator, valid_iterator, test_iterator


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        # retrieve text and no. of words
        text, text_lengths = batch.text

        predictions = model(text, text_lengths)
        loss = criterion(predictions, batch.labels.squeeze())

        acc = accuracy(predictions, batch.labels)

        # perform backpropagation
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.labels)

            acc = accuracy(predictions, batch.labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def run_train(epochs, model, train_iterator, valid_iterator, optimizer, criterion, model_type):
    best_valid_loss = float('inf')

    for epoch in range(epochs):

        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights'+'_'+model_type+'.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
=======
import torch
from torch.autograd import Variable
from torchtext import data
import matplotlib.pyplot as plt

def accuracy(probs, target):
  winners = probs.argmax(dim=1)
  corrects = (winners == target)
  accuracy = corrects.sum().float() / float(target.size(0))
  return accuracy

######################################## Using torchText ######################################

def create_iterator(train_data, valid_data, test_data, batch_size, device):
    #  BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    # by setting sort_within_batch = True.
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
        batch_size = batch_size,
        sort_key = lambda x: len(x.text), # Sort the batches by text length size
        sort_within_batch = True,
        device = device)
    return train_iterator, valid_iterator, test_iterator


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        # retrieve text and no. of words
        text, text_lengths = batch.text

        predictions = model(text, text_lengths)
        loss = criterion(predictions, batch.labels.squeeze())

        acc = accuracy(predictions, batch.labels)

        # perform backpropagation
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.labels)

            acc = accuracy(predictions, batch.labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def run_train(epochs, model, train_iterator, valid_iterator, optimizer, criterion, model_type):
    best_valid_loss = float('inf')

    for epoch in range(epochs):

        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights'+'_'+model_type+'.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
>>>>>>> 0bce7e504b2dde650f00b004789d40550baf6ac4
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')