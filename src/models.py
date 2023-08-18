from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

class MLPSTILClassifier(pl.LightningModule):
    def __init__(
            self,
            img_channels_in: int=2048,
            text_channels_in: int=1024,
            hidden_dim: int=32,
            num_classes: int=10,
            lr: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        super().__init__()
        """Initialize the classification model."""
        super().__init__()
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes) # Initialize the Accuracy object

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(img_channels_in + text_channels_in, hidden_dim)),
            # ('bn1', nn.BatchNorm1d(hidden_dim)),
            ('relu1', nn.ReLU()),
            # ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_dim, hidden_dim)),
            # ('bn2', nn.BatchNorm1d(hidden_dim)),
            ('relu2', nn.ReLU()),
            # ('dropout2', nn.Dropout(0.5)),
            ('output', nn.Linear(hidden_dim, num_classes)),
            ('softmax', nn.Softmax(dim=1))
        ]))
        
    def add_model_specific_args(parser):
        parser.add_argument("--img-channels-in", type=int)
        parser.add_argument("--text-channels-in", type=int)
        parser.add_argument("--num-classes", type=int)
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
            img_sub_x = img_sub_x.to(self.device)
            # avg wsi feats along spatial dims
            img_sub_x = img_sub_x.mean(dim=(1,2))
            # Concatenate the global image embedding with the text embedding.
            text_sub_x = text_sub_x.to(self.device).squeeze()
            mm_feats = torch.cat((img_sub_x, text_sub_x))
            out.append(mm_feats)

        out = torch.stack(out)  # Stack all tensors in the list into a single tensor
        out = self.classifier(out)  # Pass the tensor through the classifier
        # out = F.softmax(out, dim=1)  # Apply softmax along dimension 1
        return out

    def step(self, batch, batch_idx):
        img_feats, text_feats, _, stil_levels = batch
        stil_levels = stil_levels.to(self.device)
        y_hat = self.forward(img_feats, text_feats)
        # print(f"y_hat shape: {y_hat.shape}, labels shape: {labels.shape}")  # Add this line
        loss = self.loss(y_hat, stil_levels)
        return loss, y_hat, stil_levels


    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("train_loss", loss, batch_size=len(y))
        self.log("train_acc", acc, batch_size=len(y))  # Log accuracy
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("val_loss", loss, batch_size=len(y))
        self.log("val_acc", acc, batch_size=len(y))  # Log accuracy
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("test_loss", loss, batch_size=len(y))
        self.log("test_acc", acc, batch_size=len(y))  # Log accuracy
        return loss
    
    def on_train_epoch_end(self): 
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.accuracy.compute())

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
class MLPSTILRegressor(pl.LightningModule):
    def __init__(
            self,
            img_channels_in: int=2048,
            text_channels_in: int=1024,
            hidden_dim: int=32,
            lr: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        super().__init__()
        """Initialize the classification model."""
        super().__init__()
        self.save_hyperparameters()
        self.loss = nn.MSELoss()    # MSE loss fn
        # metrics
        self.train_corr = torchmetrics.PearsonCorrCoef()
        self.train_r2 = torchmetrics.R2Score()
        self.val_corr = torchmetrics.PearsonCorrCoef()
        self.val_r2 = torchmetrics.R2Score()
        self.test_corr = torchmetrics.PearsonCorrCoef()
        self.test_r2 = torchmetrics.R2Score()
        # regressor
        self.regressor = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(img_channels_in + text_channels_in, hidden_dim)),
            # ('bn1', nn.BatchNorm1d(hidden_dim)),
            ('relu1', nn.ReLU()),
            # ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_dim, hidden_dim)),
            # ('bn2', nn.BatchNorm1d(hidden_dim)),
            ('relu2', nn.ReLU()),
            # ('dropout2', nn.Dropout(0.5)),
            ('output', nn.Linear(hidden_dim, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))
        
    def add_model_specific_args(parser):
        parser.add_argument("--img-channels-in", type=int)
        parser.add_argument("--text-channels-in", type=int)
        parser.add_argument("--num-classes", type=int)
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
            img_sub_x = img_sub_x.to(self.device)
            # avg wsi feats along spatial dims
            img_sub_x = img_sub_x.mean(dim=(1,2))
            # Concatenate the global image embedding with the text embedding.
            text_sub_x = text_sub_x.to(self.device).squeeze()
            mm_feats = torch.cat((img_sub_x, text_sub_x))
            out.append(mm_feats)

        out = torch.stack(out)  # Stack all tensors in the list into a single tensor
        out = self.regressor(out)  # Pass the tensor through the regressor
        return out.squeeze()

    def step(self, batch, batch_idx):
        img_feats, text_feats, stil_scores, _ = batch
        stil_scores = stil_scores.to(self.device)
        y_hat = self.forward(img_feats, text_feats)
        loss = self.loss(y_hat, stil_scores)
        return loss, y_hat, stil_scores

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("train_loss", loss, batch_size=len(y))
        self.train_corr(y_hat, y)
        self.log("train_corr", self.train_corr)
        self.train_r2(y_hat, y)
        self.log("train_r2", self.train_r2)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("val_loss", loss, batch_size=len(y))
        self.val_corr(y_hat, y)
        self.log("val_corr", self.val_corr)
        self.val_r2(y_hat, y)
        self.log("val_r2", self.val_r2)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("test_loss", loss, batch_size=len(y))
        self.test_corr(y_hat, y)
        self.log("test_corr", self.test_corr)
        self.test_r2(y_hat, y)
        self.log("test_r2", self.test_r2)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    def on_train_epoch_end(self): 
        # log epoch metric
        self.log('train_r2_epoch', self.train_r2.compute())
        self.log('train_corr_epoch', self.train_corr.compute())

    def on_validation_epoch_end(self):
        self.log('val_r2_epoch', self.val_r2.compute())
        self.log('val_corr_epoch', self.val_corr.compute())

    def on_test_epoch_end(self):
        self.log('test_r2_epoch', self.test_r2.compute())
        self.log('test_corr_epoch', self.test_corr.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
class Attention1DSTILClassifier(pl.LightningModule):
    def __init__(
            self,
            img_channels_in: int=2048,
            text_channels_in: int=1024,
            num_classes: int=10,
            lr: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        """Initialize the classification model."""
        super().__init__()
        self.save_hyperparameters()
        self.attention = nn.Sequential(
            nn.Linear(self.hparams.img_channels_in, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hparams.img_channels_in + self.hparams.text_channels_in, self.hparams.num_classes),
            nn.Softmax(dim=1)
        )
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes) # Initialize the Accuracy object

    def add_model_specific_args(parser):
        parser.add_argument("--img-channels-in", type=int)
        parser.add_argument("--text-channels-in", type=int)
        parser.add_argument("--num-classes", type=int)
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
            img_sub_x = img_sub_x.to(self.device)
            img_sub_x = img_sub_x.reshape(self.hparams.img_channels_in, -1).T
            attention_w = self.attention(img_sub_x)
            attention_w = torch.transpose(attention_w, 1, 0)
            attention_w = nn.functional.softmax(attention_w, dim=1)
            img_sub_x = torch.mm(attention_w, img_sub_x).squeeze()

            # Concatenate the global image embedding with the text embedding.
            text_sub_x = text_sub_x.to(self.device).squeeze()
            mm_feats = torch.cat((img_sub_x, text_sub_x))

            out.append(mm_feats)

        out = torch.stack(out)  # Stack all tensors in the list into a single tensor
        out = self.classifier(out)  # Pass the tensor through the classifier
        # out = F.softmax(out, dim=1)  # Apply softmax along dimension 1

        return out

    def step(self, batch, batch_idx):
        img_feats, text_feats, _, stil_levels = batch
        stil_levels = stil_levels.to(self.device)
        y_hat = self.forward(img_feats, text_feats)
        # print(f"y_hat shape: {y_hat.shape}, labels shape: {labels.shape}")  # Add this line
        loss = self.loss(y_hat, stil_levels)
        return loss, y_hat, stil_levels


    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("train_loss", loss, batch_size=len(y))
        self.log("train_acc", acc, batch_size=len(y))  # Log accuracy
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("val_loss", loss, batch_size=len(y))
        self.log("val_acc", acc, batch_size=len(y))  # Log accuracy
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("test_loss", loss, batch_size=len(y))
        self.log("test_acc", acc, batch_size=len(y))  # Log accuracy
        return loss
    
    def on_train_epoch_end(self): 
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.accuracy.compute())

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

class Attention1DSTILRegressor(pl.LightningModule):
    def __init__(
            self,
            mode: str='multimodal',
            img_channels_in: int=2048,
            text_channels_in: int=1024,
            lr: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        """Initialize the regression model."""
        super().__init__()

        # input modalities
        assert mode in ['multimodal', 'image'], "mode must be either 'multimodal' or 'image'"
        self.mode = mode
        
        self.save_hyperparameters()

        # linear attention module
        self.attention = nn.Sequential(
            nn.Linear(img_channels_in, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # regressor
        channels_in = img_channels_in + text_channels_in if mode == 'multimodal' else img_channels_in
        self.regressor = nn.Sequential(
            nn.Linear(channels_in, 1),
            nn.Sigmoid()
        )
        
        self.loss = nn.MSELoss()    # MSE loss fn
        
        # metrics
        self.train_corr = torchmetrics.PearsonCorrCoef()
        self.train_r2 = torchmetrics.R2Score()
        self.val_corr = torchmetrics.PearsonCorrCoef()
        self.val_r2 = torchmetrics.R2Score()
        self.test_corr = torchmetrics.PearsonCorrCoef()
        self.test_r2 = torchmetrics.R2Score()
        
    def add_model_specific_args(parser):
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
            img_sub_x = img_sub_x.to(self.device)
            img_sub_x = img_sub_x.reshape(self.hparams.img_channels_in, -1).T
            attention_w = self.attention(img_sub_x)
            attention_w = torch.transpose(attention_w, 1, 0)
            attention_w = nn.functional.softmax(attention_w, dim=1)
            img_sub_x = torch.mm(attention_w, img_sub_x).squeeze()

            # Concatenate the global image embedding with the text embedding.
            text_sub_x = text_sub_x.to(self.device).squeeze()
            
            feats = torch.cat((img_sub_x, text_sub_x)) if self.mode == 'multimodal' else img_sub_x

            out.append(feats)

        out = torch.stack(out)  # Stack all tensors in the list into a single tensor
        out = self.regressor(out)  # Pass the tensor through the classifier
        # out = F.softmax(out, dim=1)  # Apply softmax along dimension 1

        return out.squeeze()

    def step(self, batch, batch_idx):
        img_feats, text_feats, stil_scores, _ = batch
        stil_scores = stil_scores.to(self.device)
        y_hat = self.forward(img_feats, text_feats)
        loss = self.loss(y_hat, stil_scores)
        return loss, y_hat, stil_scores

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("train_loss", loss, batch_size=len(y))
        self.train_corr(y_hat, y)
        self.log("train_corr", self.train_corr)
        self.train_r2(y_hat, y)
        self.log("train_r2", self.train_r2)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("val_loss", loss, batch_size=len(y))
        self.val_corr(y_hat, y)
        self.log("val_corr", self.val_corr)
        self.val_r2(y_hat, y)
        self.log("val_r2", self.val_r2)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("test_loss", loss, batch_size=len(y))
        self.test_corr(y_hat, y)
        self.log("test_corr", self.test_corr)
        self.test_r2(y_hat, y)
        self.log("test_r2", self.test_r2)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    def on_train_epoch_end(self): 
        # log epoch metric
        self.log('train_r2_epoch', self.train_r2.compute())
        self.log('train_corr_epoch', self.train_corr.compute())

    def on_validation_epoch_end(self):
        self.log('val_r2_epoch', self.val_r2.compute())
        self.log('val_corr_epoch', self.val_corr.compute())

    def on_test_epoch_end(self):
        self.log('test_r2_epoch', self.test_r2.compute())
        self.log('test_corr_epoch', self.test_corr.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
 
def pad_feature(input_tensor, dst_shape):
    width = dst_shape[-1]
    height = dst_shape[-2]
    pad_width = width - input_tensor.shape[-1]
    pad_height = height - input_tensor.shape[-2]
    padding = (0, pad_width, 0, pad_height)
    return nn.functional.pad(input_tensor, padding)
    
class EncoderDecoderAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        num_channels = 64

        # Downchanneling conv.
        self.down_channeling_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # Downsampling convs.
        self.dconv1 = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.uconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.uconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        self.merge = nn.Sequential(
            nn.Conv2d(num_channels * 2, 1, kernel_size=1, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out0 = self.down_channeling_conv(x)
        out1 = self.dconv1(out0)
        out2 = self.dconv2(out1)

        uout1 = pad_feature(self.uconv1(out2), out1.shape)
        uout1 = torch.cat((out1, uout1), 1)

        uout2 = pad_feature(self.uconv2(uout1), out0.shape)
        uout2 = torch.cat((out0, uout2), 1)

        # Final attention.
        out = self.merge(uout2)

        # Since we have variable input size, we might need to slice...
        if out.shape[2] != x.shape[2] or out.shape[3] != x.shape[3]:
            out = out[:, :, :x.shape[2], :x.shape[3]]
        return out
    
class Attention2DSTILClassifier(pl.LightningModule):
    def __init__(
            self,
            img_channels_in: int=2048,
            text_channels_in: int=1024,
            num_classes: int=10,
            lr: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        """Initialize the classification model."""
        super().__init__()
        self.save_hyperparameters()
        self.attention = EncoderDecoderAttention(img_channels_in)

        # After using attention, let's map the channels to two channels only.
        self.maps = nn.Sequential(
            nn.Conv2d(self.hparams.img_channels_in, 1, kernel_size=3, padding=1),
            #nn.BatchNorm2d(1),
            nn.ReLU(),
            #nn.Conv2d(channels_in, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(64),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hparams.img_channels_in + self.hparams.text_channels_in, self.hparams.num_classes),
            nn.Softmax(dim=1)
        )  
        
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes) # Initialize the Accuracy object

    def add_model_specific_args(parser):
        parser.add_argument("--img-channels-in", type=int)
        parser.add_argument("--text-channels-in", type=int)
        parser.add_argument("--num-classes", type=int)
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
            img_sub_x = img_sub_x.to(self.device)   # emb_dim x h x w
            attention_w = self.attention(img_sub_x.unsqueeze(0)).squeeze(0) # 1 x h x w
            # img_sub_x = self.maps(img_sub_x)  # 1 x h x w
            # take dot product of attention weights (1 x h x w) and image embeddings (emb_dim x h x w) -> emb_dim 
            img_sub_x = torch.einsum('chw,ehw->e', attention_w, img_sub_x)  # emb_dim
    
            # Concatenate the global image embedding with the text embedding.
            text_sub_x = text_sub_x.to(self.device).squeeze()
            mm_feats = torch.cat((img_sub_x, text_sub_x))
            out.append(mm_feats)

        out = torch.stack(out)  # Stack all tensors in the list into a single tensor
        out = self.classifier(out)  # Pass the tensor through the classifier
        
        return out

    def step(self, batch, batch_idx):
        img_feats, text_feats, _, stil_levels = batch
        stil_levels = stil_levels.to(self.device)
        y_hat = self.forward(img_feats, text_feats)
        # print(f"y_hat shape: {y_hat.shape}, labels shape: {labels.shape}")  # Add this line
        loss = self.loss(y_hat, stil_levels)
        return loss, y_hat, stil_levels


    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("train_loss", loss, batch_size=len(y))
        self.log("train_acc", acc, batch_size=len(y))  # Log accuracy
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("val_loss", loss, batch_size=len(y))
        self.log("val_acc", acc, batch_size=len(y))  # Log accuracy
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)  # Compute accuracy
        self.log("test_loss", loss, batch_size=len(y))
        self.log("test_acc", acc, batch_size=len(y))  # Log accuracy
        return loss
    
    def on_train_epoch_end(self): 
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.accuracy.compute())

    def on_test_epoch_end(self):
        self.log('test_acc_epoch', self.accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

class Attention2DSTILRegressor(pl.LightningModule):
    def __init__(
            self,
            img_channels_in: int=2048,
            text_channels_in: int=1024,
            lr: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        """Initialize the classification model."""
        super().__init__()
        self.save_hyperparameters()

        # attention
        self.attention = EncoderDecoderAttention(img_channels_in)

        # After using attention, let's map the channels to two channels only.
        self.maps = nn.Sequential(
            nn.Conv2d(self.hparams.img_channels_in, 1, kernel_size=3, padding=1),
            #nn.BatchNorm2d(1),
            nn.ReLU(),
            #nn.Conv2d(channels_in, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(64),
        )

        # regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.hparams.img_channels_in + self.hparams.text_channels_in, 1),
            nn.Sigmoid()
        )
        
        self.loss = nn.MSELoss()    # MSE loss fn
        
        # metrics
        self.train_corr = torchmetrics.PearsonCorrCoef()
        self.train_r2 = torchmetrics.R2Score()
        self.val_corr = torchmetrics.PearsonCorrCoef()
        self.val_r2 = torchmetrics.R2Score()
        self.test_corr = torchmetrics.PearsonCorrCoef()
        self.test_r2 = torchmetrics.R2Score()

    def add_model_specific_args(parser):
        parser.add_argument("--img-channels-in", type=int)
        parser.add_argument("--text-channels-in", type=int)
        parser.add_argument("--num-classes", type=int)
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
            img_sub_x = img_sub_x.to(self.device)   # emb_dim x h x w
            attention_w = self.attention(img_sub_x.unsqueeze(0)).squeeze(0) # 1 x h x w
            # img_sub_x = self.maps(img_sub_x)  # 1 x h x w
            # take dot product of attention weights (1 x h x w) and image embeddings (emb_dim x h x w) -> emb_dim 
            img_sub_x = torch.einsum('chw,ehw->e', attention_w, img_sub_x)  # emb_dim
    
            # Concatenate the global image embedding with the text embedding.
            text_sub_x = text_sub_x.to(self.device).squeeze()
            mm_feats = torch.cat((img_sub_x, text_sub_x))
            out.append(mm_feats)

        out = torch.stack(out)  # Stack all tensors in the list into a single tensor
        out = self.regressor(out)  # Pass the tensor through the classifier
        
        return out.squeeze()

    def step(self, batch, batch_idx):
        img_feats, text_feats, stil_scores, _ = batch
        stil_scores = stil_scores.to(self.device)
        y_hat = self.forward(img_feats, text_feats)
        loss = self.loss(y_hat, stil_scores)
        return loss, y_hat, stil_scores

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("train_loss", loss, batch_size=len(y))
        self.train_corr(y_hat, y)
        self.log("train_corr", self.train_corr)
        self.train_r2(y_hat, y)
        self.log("train_r2", self.train_r2)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("val_loss", loss, batch_size=len(y))
        self.val_corr(y_hat, y)
        self.log("val_corr", self.val_corr)
        self.val_r2(y_hat, y)
        self.log("val_r2", self.val_r2)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        self.log("test_loss", loss, batch_size=len(y))
        self.test_corr(y_hat, y)
        self.log("test_corr", self.test_corr)
        self.test_r2(y_hat, y)
        self.log("test_r2", self.test_r2)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    def on_train_epoch_end(self): 
        # log epoch metric
        self.log('train_r2_epoch', self.train_r2.compute())
        self.log('train_corr_epoch', self.train_corr.compute())

    def on_validation_epoch_end(self):
        self.log('val_r2_epoch', self.val_r2.compute())
        self.log('val_corr_epoch', self.val_corr.compute())

    def on_test_epoch_end(self):
        self.log('test_r2_epoch', self.test_r2.compute())
        self.log('test_corr_epoch', self.test_corr.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
class Attention1DSubtypeGradeClassifier(pl.LightningModule):
    def __init__(
            self,
            img_channels_in: int=2048,
            text_channels_in: int=1024,
            num_classes_region: int=4,
            num_classes_localization: int=4,
            num_classes_grade: int=4,
            lr: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        """Initialize the classification model."""
        super().__init__()
        self.save_hyperparameters()
        self.attention = nn.Sequential(
            nn.Linear(self.hparams.img_channels_in, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Modify the classifier to produce 3 outputs
        self.region_classifier = nn.Sequential(
            nn.Linear(self.hparams.img_channels_in + self.hparams.text_channels_in, num_classes_region),
            nn.Sigmoid()
        )
        self.localization_classifier = nn.Sequential(
            nn.Linear(self.hparams.img_channels_in + self.hparams.text_channels_in, num_classes_localization),
            nn.Sigmoid()
        )
        self.grade_classifier = nn.Sequential(
            nn.Linear(self.hparams.img_channels_in + self.hparams.text_channels_in, num_classes_grade),
            nn.Sigmoid()
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.region_accuracy = torchmetrics.Accuracy()
        self.localization_accuracy = torchmetrics.Accuracy()
        self.grade_accuracy = torchmetrics.Accuracy()

    def add_model_specific_args(parser):
        parser.add_argument("--img-channels-in", type=int)
        parser.add_argument("--text-channels-in", type=int)
        parser.add_argument("--num-classes", type=int)
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
            img_sub_x = img_sub_x.to(self.device)
            img_sub_x = img_sub_x.reshape(self.hparams.img_channels_in, -1).T
            attention_w = self.attention(img_sub_x)
            attention_w = torch.transpose(attention_w, 1, 0)
            attention_w = nn.functional.softmax(attention_w, dim=1)
            img_sub_x = torch.mm(attention_w, img_sub_x).squeeze()

            # Concatenate the global image embedding with the text embedding.
            text_sub_x = text_sub_x.to(self.device).squeeze()
            mm_feats = torch.cat((img_sub_x, text_sub_x))

            out.append(mm_feats)

        out = torch.stack(out)  # Stack all tensors in the list into a single tensor
        # Pass the tensor through the classifier
        region_out = self.region_classifier(out)
        localization_out = self.localization_classifier(out)
        grade_out = self.grade_classifier(out)
        return region_out, localization_out, grade_out
        # out = F.softmax(out, dim=1)  # Apply softmax along dimension 1

        return out

    def step(self, batch, batch_idx):
        img_feats, text_feats, region_labels, localization_labels, grade_labels = batch
        region_labels, localization_labels, grade_labels = region_labels.to(self.device), localization_labels.to(self.device), grade_labels.to(self.device)
        region_out, localization_out, grade_out = self.forward(img_feats, text_feats)
        
        loss_region = self.loss(region_out, region_labels)
        loss_localization = self.loss(localization_out, localization_labels)
        loss_grade = self.loss(grade_out, grade_labels)
        
        total_loss = loss_region + loss_localization + loss_grade
        return total_loss, region_out, localization_out, grade_out, region_labels, localization_labels, grade_labels

    def training_step(self, batch, batch_idx):
        loss, region_out, localization_out, grade_out, region_labels, localization_labels, grade_labels = self.step(batch, batch_idx)
        acc_region = self.region_accuracy(region_out, region_labels)
        acc_localization = self.localization_accuracy(localization_out, localization_labels)
        acc_grade = self.grade_accuracy(grade_out, grade_labels)
        
        self.log("train_loss", loss, batch_size=len(region_labels))
        self.log("train_acc_region", acc_region, batch_size=len(region_labels))
        self.log("train_acc_localization", acc_localization, batch_size=len(localization_labels))
        self.log("train_acc_grade", acc_grade, batch_size=len(grade_labels))
        return loss

    # Similarly, update the validation_step and test_step functions
    def validation_step(self, batch, batch_idx):
            
        self.log("val_loss", loss, batch_size=len(region_labels))
        self.log("val_acc_region", acc_region, batch_size=len(region_labels))
        self.log("val_acc_localization", acc_localization, batch_size=len(localization_labels))
        self.log("val_acc_grade", acc_grade, batch_size=len(grade_labels))
        return loss        

    def test_step(self, batch, batch_idx):
        loss, region_out, localization_out, grade_out, region_labels, localization_labels, grade_labels = self.step(batch, batch_idx)
        acc_region = self.region_accuracy(region_out, region_labels)
        acc_localization = self.localization_accuracy(localization_out, localization_labels)
        acc_grade = self.grade_accuracy(grade_out, grade_labels)

        self.log("test_loss", loss, batch_size=len(region_labels))
        self.log("test_acc_region", acc_region, batch_size=len(region_labels))
        self.log("test_acc_localization", acc_localization, batch_size=len(localization_labels))
        self.log("test_acc_grade", acc_grade, batch_size=len(grade_labels))
        return loss
    
    def on_train_epoch_end(self): 
        # log epoch metric
        self.log('train_acc_region_epoch', self.region_accuracy.compute())
        self.log('train_acc_localization_epoch', self.localization_accuracy.compute())
        self.log('train_acc_grade_epoch', self.grade_accuracy.compute())

    def on_validation_epoch_end(self):
        self.log('val_acc_region_epoch', self.region_accuracy.compute())
        self.log('val_acc_localization_epoch', self.localization_accuracy.compute())
        self.log('val_acc_grade_epoch', self.grade_accuracy.compute())

    def on_test_epoch_end(self):
        self.log('test_acc_region_epoch', self.region_accuracy.compute())
        self.log('test_acc_localization_epoch', self.localization_accuracy.compute())
        self.log('test_acc_grade_epoch', self.grade_accuracy.compute())
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer   