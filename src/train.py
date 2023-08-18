import os
from tqdm import tqdm
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from .utils import *

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

def train_mm_stil(model, train_loader, val_loader, args):
    """
    Trains the given multimodal STIL classifier model with the provided arguments.

    Args:
        model (pl.LightningModule): The PyTorch Lightning model to be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): The DataLoader for the validation data.
        args (dict): A dictionary containing the following keys:
            - num_epochs (int, optional): The number of epochs for training. Defaults to 10.
            - patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 5.
            - ckpt_dir (str, optional): The directory where the model checkpoints will be saved. Defaults to 'model_ckpts/'.
            - ckpt_name (str, optional): The name of the model checkpoint file. Defaults to 'ckpt_best'.
            - resume_ckpt (str, optional): The path to a checkpoint from which training will resume. Defaults to None.
    """

    # Define early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.get('patience', 5),
        verbose=True,
        mode='min'
    )

    # Define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath= os.path.join(args.get('ckpt_dir', 'model_ckpts'), model.__class__.__name__),
        filename=args.get('ckpt_name', 'ckpt_best'),
        save_top_k=1,
        mode='min'
    )
    
    # Define logger
    # Define logger
    logger = TensorBoardLogger("lightning_logs", name=model.__class__.__name__)


    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=args.get('num_epochs', 10),
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=5,
        logger=logger,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.get('resume_ckpt', None))
    print(f'training on device: {model.device}')
    return model, trainer

def kfold_cv(model_class, dataset, model_args={}, train_args={}):
    """
    Performs k-fold cross-validation training.

    Args:
        model: the model to be trained.
        dataset: The entire dataset.
        model_args (dict): model args
        train_args (dict): training args
    """
    # Initialize KFold
    k = train_args.get('k', 5)    # number of folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Store results for each fold
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset.df)):
        print(f"training fold {fold + 1}/{k}")

        # Create data loaders for the current fold
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=train_args.get('bsz', 32), shuffle=True, collate_fn=MMSTILDataset.mm_collate_fn, num_workers=12)
        val_loader = DataLoader(val_subset, batch_size=train_args.get('bsz', 32), shuffle=False, collate_fn=MMSTILDataset.mm_collate_fn, num_workers=12)

        # Initialize model and train for the current fold
        model = model_class(**model_args)
        model, trainer = train_mm_stil(model, train_loader, val_loader, train_args)

        # Evaluate the trained model on the validation set for the current fold
        results = trainer.test(model, val_loader)
        fold_results.append(results[0])
    
    # Write results to log file
    log_file = f'outputs/{model_class.__name__}/kfold_cv.json' if not train_args.get('log_file', None) else train_args['log_file']
    os.makedirs(log_file.rsplit('/', 1)[0], exist_ok=True)
    log = {
        'model': model_class.__name__,
        'k': k,
        'args': train_args,
        'results': fold_results
    }
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=4)

    return fold_results
