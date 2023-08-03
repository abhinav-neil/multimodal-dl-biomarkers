#!/usr/bin/env python3
"""Preprocess a given list of slide images."""

import pathlib
from typing import List, TextIO, Callable, Optional, Any
import sys
from enum import Enum

import torch
from torch.utils.data import DataLoader
import dlup
from dlup.data.dataset import TiledROIsSlideImageDataset as DLUPDataset
import dlup.tiling
from dlup.experimental_backends import ImageBackend
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, track

# sys.path.append('/home/neil/multimodal/stils/src/stils')
from models import retccl_resnet50, HistoRetCCLResnet50_Weights


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
        invalid_slides_path: pathlib.Path = typer.Option('data/stils/invalid_slides.txt', help="File to write invalid slides to."),
        slides_manifest_file_path: pathlib.Path = typer.Argument(
            ..., help="File from which to read the slides paths to be processed."),
        output_dir_path: pathlib.Path = typer.Argument(..., help="Output directory path in which to store the features."),
):
    """Process slides from a manifest containing their paths and save the features in a folder."""
    is_stdin = str(slides_manifest_file_path) == '-'
    input_file_object = sys.stdin if is_stdin else open(slides_manifest_file_path, 'r')
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
    
    output_dir_path.mkdir(exist_ok=True)
    encoder.cuda().eval()
    with torch.no_grad():
        for i, slide_path in enumerate(slides_paths):
            try:
                slide_dataset = slide_dataset_factory(slide_path)
                process_slide(
                    encoder,
                    num_filters,
                    batch_size,
                    output_dir_path,
                    slide_dataset,
                    i,
                    len(slides_paths),
                    skip_existing=skip_existing_outputs
                )
            except dlup.UnsupportedSlideError:
                print(f"invalid file: {slide_path}")
                # write unsupported file name to file
                with open(invalid_slides_path, 'a') as f:
                    f.write(str(slide_path) + '\n')


if __name__ == "__main__":
    typer.run(main)
