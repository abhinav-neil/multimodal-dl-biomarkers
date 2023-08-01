from tqdm import tqdm
import torch
import torch.nn as nn

def train_mm(model, train_loader, val_loader, epochs, device, patience=5):
    """
    Trains a model for a specified number of epochs and evaluates it on a validation set.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): The DataLoader for the training data.
        val_dataloader (torch.utils.data.DataLoader): The DataLoader for the validation data.
        epochs (int): The number of epochs to train for.
        device (torch.device): The device (CPU or GPU) where the model should be trained.
        patience (int): The number of epochs to wait for improvement in the validation loss before stopping training.

    Returns:
        model (torch.nn.Module): The trained model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(epochs)):
        print(f'epoch: {epoch + 1} / {epochs}')
        # train
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss_epoch = running_loss / len(train_loader.dataset)
        # val
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                correct_predictions += torch.sum(preds == labels.data)
            val_loss_epoch = running_loss / len(val_loader.dataset)
            epoch_acc = correct_predictions.double() / len(val_loader.dataset)
            
        print(f'train loss: {train_loss_epoch:.3f}, val loss: {val_loss_epoch:.3f}, val acc: {epoch_acc:.3f}')
        
        # check for improvement in validation loss
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'early stopping after {patience} epochs without improvement in validation loss')
                break
            
    return model

def evaluate(model, test_loader, device):
    """
    Evaluates a trained model on a test set.

    Args:
        model (torch.nn.Module): The trained model.
        test_dataloader (torch.utils.data.DataLoader): The DataLoader for the test data.
        device (torch.device): The device (CPU or GPU) where the model should be evaluated.

    Returns:
        test_loss (float): The loss of the model on the test set.
        test_acc (float): The accuracy of the model on the test set.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct_predictions.double() / len(test_loader.dataset)

    print(f'test Loss: {test_loss:.3f}, test Acc: {test_acc:.3f}')

    return test_loss, test_acc