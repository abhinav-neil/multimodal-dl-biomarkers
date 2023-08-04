from collections import OrderedDict
from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision.models.resnet import conv3x3, conv1x1, BasicBlock
from torch.nn.parameter import Parameter
from torchvision.models.resnet import Weights, WeightsEnum
from torchvision import transforms
import pytorch_lightning as pl

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            momentum_bn: float = 0.1
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, momentum=momentum_bn)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, momentum=momentum_bn)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, momentum=momentum_bn)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class RetCCLResNet(nn.Module):

    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            two_branch=False,
            mlp=False,
            normlinear=False,
            momentum_bn=0.1,
            attention=False,
            attention_layers=3,
            return_attn=False
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.return_attn = return_attn
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.two_branch = two_branch
        self.momentum_bn = momentum_bn
        self.mlp = mlp
        linear = NormedLinear if normlinear else nn.Linear

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes, momentum=momentum_bn)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        if attention:
            self.att_branch = self._make_layer(block, 512, attention_layers, 1, attention=True)
        else:
            self.att_branch = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.mlp:
            if self.two_branch:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512 * block.expansion),
                    nn.ReLU()
                )
                self.instDis = linear(512 * block.expansion, num_classes)
                self.groupDis = linear(512 * block.expansion, num_classes)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512 * block.expansion),
                    nn.ReLU(),
                    linear(512 * block.expansion, num_classes)
                )
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            if self.two_branch:
                self.groupDis = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, attention=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, momentum=self.momentum_bn),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, momentum_bn=self.momentum_bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, momentum_bn=self.momentum_bn))

        if attention:
            layers.append(nn.Sequential(
                conv1x1(self.inplanes, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                conv1x1(128, 1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.att_branch is not None:
            att_map = self.att_branch(x)
            x = x + att_map * x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.mlp and self.two_branch:
            x = self.fc(x)
            x1 = self.instDis(x)
            x2 = self.groupDis(x)
            return [x1, x2]
        else:
            x1 = self.fc(x)
            if self.two_branch:
                x2 = self.groupDis(x)
                return [x1, x2]
            return x1
        
        
class HistoRetCCLResnet50_Weights(WeightsEnum):
    """Weights adapted from: https://github.com/Xiyue-Wang/RetCCL.

    Original input size is 256 at 1mpp.
    """
    RetCCLWeights = Weights(
        url='https://storage.googleapis.com/cold.s3.ellogon.ai/resnet50-histo-retccl.pth',
        transforms=transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]),
        meta={}
    )
    
def retccl_resnet50(*, weights: Optional[HistoRetCCLResnet50_Weights] = None,
             progress: bool = True, **kwargs: Any) -> RetCCLResNet:
    model = RetCCLResNet(
        Bottleneck,
        [3, 4, 6, 3],
        # num_classes=128,
        # mlp=False,
        # two_branch=False,
        # normlinear=True,
        **kwargs
    )
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
    return model

class MLPSTILClassifier(pl.LightningModule):
    def __init__(
            self,
            img_channels_in: int,
            text_channels_in: int,
            hidden_dim: int,
            num_classes: int,
            learning_rate: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        super().__init__()
        """Initialize the classification model."""
        super().__init__()
        self.save_hyperparameters()
        self.loss = torch.nn.CrossEntropyLoss()
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
        parser.add_argument("--learning_rate", type=float, default=5e-4)
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
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
class MLPSTILRegressor(pl.LightningModule):
    def __init__(
            self,
            img_channels_in: int,
            text_channels_in: int,
            hidden_dim: int,
            learning_rate: float = 5e-4,
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
        parser.add_argument("--learning_rate", type=float, default=5e-4)
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
            lr=self.hparams.learning_rate,
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
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
class Attention1DSTILClassifier(pl.LightningModule):
    def __init__(
            self,
            img_channels_in: int,
            text_channels_in: int,
            num_classes: int,
            learning_rate: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        """Initialize the classification model."""
        super().__init__()
        self.save_hyperparameters()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.img_channels_in, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.img_channels_in + self.hparams.text_channels_in, self.hparams.num_classes),
            torch.nn.Softmax(dim=1)
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes) # Initialize the Accuracy object

    def add_model_specific_args(parser):
        parser.add_argument("--img-channels-in", type=int)
        parser.add_argument("--text-channels-in", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--learning_rate", type=float, default=5e-4)
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
            attention_w = torch.nn.functional.softmax(attention_w, dim=1)
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
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

class Attention1DSTILRegressor(pl.LightningModule):
    def __init__(
            self,
            img_channels_in: int,
            text_channels_in: int,
            learning_rate: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        """Initialize the classification model."""
        super().__init__()
        self.save_hyperparameters()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.img_channels_in, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1),
        )
        
        # regressor
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.img_channels_in + self.hparams.text_channels_in, 1),
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
        parser.add_argument("--learning_rate", type=float, default=5e-4)
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
            attention_w = torch.nn.functional.softmax(attention_w, dim=1)
            img_sub_x = torch.mm(attention_w, img_sub_x).squeeze()

            # Concatenate the global image embedding with the text embedding.
            text_sub_x = text_sub_x.to(self.device).squeeze()
            mm_feats = torch.cat((img_sub_x, text_sub_x))

            out.append(mm_feats)

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
            lr=self.hparams.learning_rate,
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
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
 
def pad_feature(input_tensor, dst_shape):
    width = dst_shape[-1]
    height = dst_shape[-2]
    pad_width = width - input_tensor.shape[-1]
    pad_height = height - input_tensor.shape[-2]
    padding = (0, pad_width, 0, pad_height)
    return torch.nn.functional.pad(input_tensor, padding)
    
class EncoderDecoderAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        num_channels = 64

        # Downchanneling conv.
        self.down_channeling_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_channels),
            torch.nn.ReLU()
        )

        # Downsampling convs.
        self.dconv1 = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.dconv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.uconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.uconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(num_channels),
            torch.nn.ReLU()
        )
        self.merge = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels * 2, 1, kernel_size=1, stride=1, padding=1),
            torch.nn.Sigmoid()
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
            img_channels_in: int,
            text_channels_in: int,
            num_classes: int,
            learning_rate: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        """Initialize the classification model."""
        super().__init__()
        self.save_hyperparameters()
        self.attention = EncoderDecoderAttention(img_channels_in)

        # After using attention, let's map the channels to two channels only.
        self.maps = torch.nn.Sequential(
            torch.nn.Conv2d(self.hparams.img_channels_in, 1, kernel_size=3, padding=1),
            #torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(),
            #torch.nn.Conv2d(channels_in, 64, kernel_size=3, padding=1),
            #torch.nn.BatchNorm2d(64),
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.img_channels_in + self.hparams.text_channels_in, self.hparams.num_classes),
            torch.nn.Softmax(dim=1)
        )  
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes) # Initialize the Accuracy object

    def add_model_specific_args(parser):
        parser.add_argument("--img-channels-in", type=int)
        parser.add_argument("--text-channels-in", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--learning_rate", type=float, default=5e-4)
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
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

class Attention2DSTILRegressor(pl.LightningModule):
    def __init__(
            self,
            img_channels_in: int,
            text_channels_in: int,
            learning_rate: float = 5e-4,
            weight_decay: float = 5e-4,
            **kwargs,
    ):
        """Initialize the classification model."""
        super().__init__()
        self.save_hyperparameters()

        # attention
        self.attention = EncoderDecoderAttention(img_channels_in)

        # After using attention, let's map the channels to two channels only.
        self.maps = torch.nn.Sequential(
            torch.nn.Conv2d(self.hparams.img_channels_in, 1, kernel_size=3, padding=1),
            #torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(),
            #torch.nn.Conv2d(channels_in, 64, kernel_size=3, padding=1),
            #torch.nn.BatchNorm2d(64),
        )

        # regressor
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.hparams.img_channels_in + self.hparams.text_channels_in, 1),
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
        parser.add_argument("--learning_rate", type=float, default=5e-4)
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
            lr=self.hparams.learning_rate,
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
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    