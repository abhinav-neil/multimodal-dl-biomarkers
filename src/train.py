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

def train_mm(model, train_loader, val_loader, args):
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
            - tblog_name (str, optional): The name of the TensorBoard log. Defaults to 'last'.
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
        dirpath= os.path.join(args.get('ckpt_dir', 'model_ckpts'), model.__class__.__name__, model.hparams.target),
        filename=args.get('ckpt_name', 'ckpt_best'),
        save_top_k=args.get('save_top_k', 1),
        mode='min'
    )
    
    # Define logger
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=os.path.join(model.__class__.__name__, model.hparams.target),
        version=args.get('tblog_name', 'last')
    )


    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=args.get('num_epochs', 50),
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=5,
        logger=logger,
        enable_progress_bar=args.get('enable_progress_bar', True),
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
        train_loader = DataLoader(train_subset, batch_size=train_args.get('bsz', 32), shuffle=True, collate_fn=mm_collate_fn, num_workers=12)
        val_loader = DataLoader(val_subset, batch_size=train_args.get('bsz', 32), shuffle=False, collate_fn=mm_collate_fn, num_workers=12)

        # Initialize model and train for the current fold
        model = model_class(**model_args)
        model, trainer = train_mm(model, train_loader, val_loader, train_args)

        # Evaluate the trained model on the validation set for the current fold
        results = trainer.test(model, val_loader)
        fold_results.append(results[0])
    
    # Write results to log file
    if train_args.get('log_file', None):
        log_file = train_args['log_file']
    else:
        log_file = f'outputs/{model_class.__name__}/{model.hparams.target}/kfold_{model.hparams.mode}.json'
    os.makedirs(log_file.rsplit('/', 1)[0], exist_ok=True)
    log = {
        'model': model_class.__name__,
        'k': k,
        'model_args': model_args,
        'train_args': train_args,
        'results': fold_results
    }
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=4)

    return fold_results
