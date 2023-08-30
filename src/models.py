import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
  
class Attention1DRegressor(pl.LightningModule):
    '''
    Attention-based regression model for STIL/MSI prediction.
    '''
    def __init__(
            self,
            mode: str='mm',
            target: str='stils',
            img_channels_in: int=2048,
            text_channels_in: int=1024,
            lr: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        '''
        Initialize the 1D attention regressor.add()
        Inputs:
            mode: str, input modality, either 'mm' or 'img' or 'text'
            target: str, target variable, either 'stils' or 'msi'
            img_channels_in: int, dim of image features
            text_channels_in: int, dim of text features
            lr: float, learning rate for optimizer
            weight_decay: float, weight decay for optimizer
        '''
        super().__init__()
        # target variable
        assert target in ['stils', 'msi'], "target must be either 'stils' or 'msi'"
        # input modalities
        assert mode in ['mm', 'img', 'text'], "mode must be either 'mm' or 'img' or 'text'"
        self.save_hyperparameters()

        # linear attention module
        self.attention = nn.Sequential(
            nn.Linear(img_channels_in, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # regressor
        channels_in = img_channels_in + text_channels_in if mode == 'mm' else img_channels_in if mode == 'img' else text_channels_in
        self.regressor = nn.Sequential(
            nn.Linear(channels_in, 1),
            nn.Sigmoid()
        )
        
        self.loss = nn.MSELoss()    # MSE loss fn
        
        # metrics
        self.corr = torchmetrics.PearsonCorrCoef()
        self.r2 = torchmetrics.R2Score()
        
    def add_model_specific_args(parser):
        parser.add_argument("--mode", type=str),
        parser.add_argument("--target", type=str),
        parser.add_argument("--img-channels-in", type=int)
        parser.add_argument("--text-channels-in", type=int)
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--weight_decay", type=float, default=5e-4)
        return parser

    def forward(self, img_feats, text_feats):
        """Model forward.
        We expect a list of tensors for both image and text features. 
        Each list item tensor contains the features of a single slide.
        """
        out = []
        for img_sub_x, text_sub_x in zip(img_feats, text_feats):
            # Weighted sum of all the image feature vectors.
            img_sub_x = img_sub_x.reshape(self.hparams.img_channels_in, -1).T
            attention_w = self.attention(img_sub_x)
            attention_w = torch.transpose(attention_w, 1, 0)
            attention_w = nn.functional.softmax(attention_w, dim=1)
            img_sub_x = torch.mm(attention_w, img_sub_x).squeeze()

            # Concatenate the global image embedding with the text embedding.
            text_sub_x = text_sub_x.squeeze()
            
            feats = torch.cat((img_sub_x, text_sub_x)) if self.hparams.mode == 'mm' else img_sub_x

            out.append(feats)

        out = torch.stack(out)  # Stack all tensors in the list into a single tensor
        out = self.regressor(out)  # Pass the tensor through the classifier

        return out.squeeze()

    def step(self, batch, batch_idx):
        img_feats, text_feats = batch['wsi_feats'], batch['report_feats']
        labels = batch['stil_score'] if self.hparams.target == 'stils' else batch['msi_score']
        out = self.forward(img_feats, text_feats)
        loss = self.loss(out, labels)
        return loss, out, labels

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=len(y))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, batch_size=len(y))
        self.log("val_corr", self.corr(y_hat, y), on_step=True, on_epoch=True, batch_size=len(y))
        self.log("val_r2", self.r2(y_hat, y), on_step=True, on_epoch=True, batch_size=len(y))
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("test_corr", self.corr(y_hat, y), on_step=True, on_epoch=True, batch_size=len(y))
        self.log("test_r2", self.r2(y_hat, y), on_step=True, on_epoch=True, batch_size=len(y))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
class Attention1DClassifier(pl.LightningModule):
    def __init__(
            self,
            mode: str='mm',
            target: str='region',
            img_channels_in: int=2048,
            text_channels_in: int=1024,
            num_classes: int=4,
            lr: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        '''
        Initialize the 1D attention classifier.
        Inputs:
            mode: str, input modality, either 'mm' or 'img'
            target: str, target variable ('region' or 'local' or 'grade')
            img_channels_in: int, dim of image features
            text_channels_in: int, dim of text features
            num_classes: int, number of target classes
            lr: float, learning rate for optimizer
            weight_decay: float, weight decay for optimizer
        '''
        super().__init__()
        
        # input modalities
        assert mode in ['mm', 'text', 'img'], "mode must be either 'mm' or 'text' or 'img'"
        # target
        assert target in ['region', 'local', 'grade'], "target must be either 'region' or 'local' or 'grade'"
        
        self.save_hyperparameters()
        self.attention = nn.Sequential(
            nn.Linear(img_channels_in, 128),
            # nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # Modify the classifier to produce 3 outputs
        channels_in = img_channels_in + text_channels_in if mode == 'mm' else img_channels_in if mode == 'img' else text_channels_in
        
        # initialize classifiers
        self.classifier = nn.Sequential(
            nn.Linear(channels_in, num_classes))
        
        self.loss = nn.CrossEntropyLoss()   # define cross-entropy loss
        
        # metrics
        self.acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def add_model_specific_args(parser):
        parser.add_argument("--mode", type=str),
        parser.add_argument("--target", type=str),
        parser.add_argument("--img-channels-in", type=int)
        parser.add_argument("--text-channels-in", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--weight_decay", type=float, default=5e-4)
        return parser

    def forward(self, img_feats, text_feats):
        """Forward pass of the model.
        Inputs:
            img_feats: list of tensors containing the image features of each slide
            text_feats: list of tensors containing the text features of each slide
        Returns:
            out: tensor containing the classification output for each sample
        """
        out = []
        for img_sub_x, text_sub_x in zip(img_feats, text_feats):
            # get text features
            text_sub_x = text_sub_x.to(self.device).squeeze()   

            if self.hparams.mode == 'text':
                out.append(text_sub_x)
            else:
                # Weighted sum of all the image feature vectors.
                img_sub_x = img_sub_x.to(self.device)
                img_sub_x = img_sub_x.reshape(self.hparams.img_channels_in, -1).T
                attention_w = self.attention(img_sub_x)
                attention_w = torch.transpose(attention_w, 1, 0)
                attention_w = F.softmax(attention_w, dim=1)
                img_sub_x = torch.mm(attention_w, img_sub_x).squeeze()

                # Concatenate the global image embedding with the text embedding.
                feats = torch.cat((img_sub_x, text_sub_x)) if self.hparams.mode == 'mm' else img_sub_x
                out.append(feats)

        out = torch.stack(out)  # Stack all tensors in the list into a single tensor
        # Pass the tensor through the classifier
        out = self.classifier(out)
        
        return out

    def step(self, batch, batch_idx):
        img_feats, text_feats, labels = batch['wsi_feats'], batch['report_feats'], batch[self.hparams.target]
        out = self.forward(img_feats, text_feats)
        loss = self.loss(out, labels)
        return loss, out, labels

    def training_step(self, batch, batch_idx):
        loss, out, labels = self.step(batch, batch_idx)
        self.log("train_loss", loss, batch_size=len(labels), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, out, labels = self.step(batch, batch_idx)
        acc = self.acc(out, labels)
        self.log("val_loss", loss, batch_size=len(labels), on_step=True, on_epoch=True)
        self.log("val_acc", acc, batch_size=len(labels), on_step=True, on_epoch=True)
        return loss        

    def test_step(self, batch, batch_idx):
        loss, out, labels = self.step(batch, batch_idx)
        acc = self.acc(out, labels)
        self.log("test_loss", loss, batch_size=len(labels), on_step=True, on_epoch=True)
        self.log("test_acc", acc, batch_size=len(labels), on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer   