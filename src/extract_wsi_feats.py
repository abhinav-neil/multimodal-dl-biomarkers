"""Preprocess a given list of slide images."""

import pathlib
from typing import List, TextIO, Callable, Optional, Any
import sys
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3, conv1x1, BasicBlock
from torch.nn.parameter import Parameter
from torchvision.models.resnet import Weights, WeightsEnum
from torchvision import transforms
from torch.utils.data import DataLoader
import dlup
from dlup.data.dataset import TiledROIsSlideImageDataset as DLUPDataset
import dlup.tiling
from dlup.experimental_backends import ImageBackend
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, track

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

def is_valid_slide_path(slide_path: pathlib.Path, skip_invalid: bool = False):
    """Return true if the path points to a valid slide path."""
    slide_image = dlup.SlideImage.from_file_path(slide_path, ImageBackend.OPENSLIDE)
    if slide_image.mpp > 1e2:
        if skip_invalid:
            return False
        raise ValueError('slide base mpp greater than 100')
    return True


def parse_manifest(
        input_file_object: TextIO,
        slides_root_dir: pathlib.Path,
        skip_invalid_slides: bool = False,
        skip_slides_checks: bool = False,
) -> List[pathlib.Path]:
    """The manifest file is expected to be a list of file paths."""
    manifest_paths = [pathlib.Path(line.strip()) for line in input_file_object]
    if slides_root_dir is not None:
        manifest_paths = [slides_root_dir / path for path in manifest_paths]
    if skip_slides_checks:
        return manifest_paths
    paths = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        task_id = progress.add_task(description="", total=None)
        for path in manifest_paths:
            progress.update(
                task_id,
                description=f"Checking slide {str(path.name)}...", refresh=True
            )
            if is_valid_slide_path(path, skip_invalid=skip_invalid_slides):
                paths.append(path)
    return list(paths)


class DLUPDatasetWrapper(DLUPDataset):
    """Thin wrapper to return plain RGB images."""

    def __init__(self, *args, transform: Callable | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out = out['image'].convert('RGB')
        if self.transform:
            return self.transform(out)
        return out


def process_slide(
        model: torch.nn.Module,
        num_filters: int,
        batch_size: int,
        output_directory_path: pathlib.Path,
        slide_dataset: DLUPDatasetWrapper,
        slide_number: int,
        total_slides: int,
        skip_existing: int,
        num_workers=8,
):
    # Create the output tensor
    grid = slide_dataset.grids[0][0]
    width, height = grid.size

    # We expect to have a tensor of size num_filters, *output_size
    output = torch.empty((num_filters, height, width), dtype=torch.float32)
    train_loader = DataLoader(slide_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    output_file_path = output_directory_path / (slide_dataset.path.stem + '.wsi.pt')
    if skip_existing and output_file_path.exists():
        return

    batch_number = 0
    for batch in track(
            train_loader,
            description=f"Processing ({slide_number}/{total_slides}) {slide_dataset.path.stem}:"
    ):
        batch = batch.cuda()
        out = model(batch).reshape(-1, num_filters)
        start_index = batch_number * batch_size
        output.view(num_filters, -1)[:, start_index:start_index + len(out)] = out.T
        batch_number += 1
    torch.save(output, output_file_path)


def get_dataset_factory(mpp: float, tile_size: int, transform: Callable):
    def dataset_factory(slide_path: pathlib.Path) -> DLUPDatasetWrapper:
        return DLUPDatasetWrapper.from_standard_tiling(
            slide_path,
            mpp,
            tile_size=(tile_size, tile_size),
            tile_overlap=(0, 0),
            tile_mode=dlup.tiling.TilingMode.skip,
            grid_order=dlup.tiling.GridOrder.C,
            crop=False,
            transform=transform,
            backend=ImageBackend.OPENSLIDE
        )
    return dataset_factory

def retccl_resnet50_factory(mpp: Optional[float], tile_size: Optional[int]):
    model = retccl_resnet50(
        weights=HistoRetCCLResnet50_Weights.RetCCLWeights,
        num_classes=128,
        mlp=False,
        two_branch=False,
        normlinear=True
    )
    transforms = HistoRetCCLResnet50_Weights.RetCCLWeights.value.transforms
    num_filters = model.fc.in_features
    model.fc = torch.nn.Identity()
    final_mpp = 1.0 if mpp is None else mpp
    final_tile_size = 256 if tile_size is None else tile_size
    return model, get_dataset_factory(final_mpp, final_tile_size, transforms), num_filters


class PretrainedModel(str, Enum):
    RETCCL = "retccl"


def main(
        pretrained_model: PretrainedModel = PretrainedModel.RETCCL,
        tile_size: Optional[int] = typer.Option(512, help="Override default tile size to use to process the slides."),
        mpp: Optional[float] = typer.Option(1., help="Override default resolution in microns per pixel."),
        batch_size: int = typer.Option(256, help="Batch size to use."),
        skip_invalid_slides: bool = typer.Option(True, help="Skip without complaints if invalid slides are found."),
        skip_existing_outputs: bool = typer.Option(True, help="If an output already exists, skip the slide."),
        skip_slides_checks: bool = typer.Option(True, help="Do not validate the file paths before processing."),
        slides_root_dir: pathlib.Path = typer.Option(..., help="Root directory in which to find the slides."),
        invalid_slides_path: Optional[pathlib.Path] = typer.Option(None, help="File to write invalid slides to."),
        slides_manifest_path: pathlib.Path = typer.Option(..., help="File from which to read the slides paths to be processed."),
        output_dir: pathlib.Path = typer.Option(..., help="Output directory path in which to store the features."),
):
    """Process slides from a manifest containing their paths and save the features in a folder."""
    is_stdin = str(slides_manifest_path) == '-'
    input_file_object = sys.stdin if is_stdin else open(slides_manifest_path, 'r')
    slides_paths = parse_manifest(
        input_file_object,
        slides_root_dir,
        skip_invalid_slides=skip_invalid_slides,
        skip_slides_checks=skip_slides_checks
    )

    encoder, slide_dataset_factory, num_filters = {
        PretrainedModel.RETCCL: retccl_resnet50_factory
    }[pretrained_model](
        mpp=mpp,
        tile_size=tile_size
    )
    
    output_dir.mkdir(exist_ok=True)
    encoder.cuda().eval()
    with torch.no_grad():
        for i, slide_path in enumerate(slides_paths):
            try:
                slide_dataset = slide_dataset_factory(slide_path)
                process_slide(
                    encoder,
                    num_filters,
                    batch_size,
                    output_dir,
                    slide_dataset,
                    i,
                    len(slides_paths),
                    skip_existing=skip_existing_outputs
                )
            except dlup.UnsupportedSlideError as e:
                print(f"encountered error {e} when processing {slide_path}")
                if invalid_slides_path is not None:
                    # write unsupported file name to file
                    with open(invalid_slides_path, 'a') as f:
                        f.write(str(slide_path) + '\n')


if __name__ == "__main__":
    typer.run(main)
