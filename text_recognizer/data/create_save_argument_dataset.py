import sys
sys.path.append("D:/RL_Finance/Image_to_Text")


import text_recognizer.metadata.iam as metadata_iam
from text_recognizer import util
from IPython.display import Image
from text_recognizer.data.iam import IAM
from text_recognizer.data.util import resize_image
from typing import Callable, List, Sequence, Tuple
import numpy as np
import torch, json
import random
from pathlib import Path
from PIL import Image
import text_recognizer.metadata.iam_lines as metadata_iam_lines
import text_recognizer.metadata.iam_paragraphs as metadata_iam_paragraphs
from text_recognizer.data.util import convert_strings_to_labels
from text_recognizer.stems.paragraph import ParagraphStem
import argparse


IMAGE_SCALE_FACTOR = metadata_iam_lines.IMAGE_SCALE_FACTOR
DL_DATA_DIRNAME = metadata_iam.DL_DATA_DIRNAME
mapping = metadata_iam_paragraphs.MAPPING
inverse_mapping = {v: k for k, v in enumerate(mapping)}

iam=IAM()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training dataset for ResNet-Transformer.")
    
    parser.add_argument("--dataset_len", type=int, default=50, help="Number of samples in the dataset")
    parser.add_argument("--input_dims", type=int, nargs='+', default=metadata_iam_paragraphs.DIMS, help="Input dimensions of the images")
    parser.add_argument("--output_dims", type=int, nargs='+', default=metadata_iam_paragraphs.OUTPUT_DIMS, help="Output dimensions (number of classes)")
    
    args = parser.parse_args()
    return args

def generate_line_crops_and_labels(iam: IAM, split: str, scale_factor=IMAGE_SCALE_FACTOR):
    """Create both cropped lines and associated labels from IAM, with resizing by default"""
    crops, labels = [], []
    for iam_id in iam.ids_by_split[split]:
        labels += iam.line_strings_by_id[iam_id]

        image = iam.load_image(iam_id)
        for line in iam.line_regions_by_id[iam_id]:
            coords = [line[point] for point in ["x1", "y1", "x2", "y2"]]
            crop = image.crop(coords)
            crop = resize_image(crop, scale_factor=scale_factor)
            crops.append(crop)

    assert len(crops) == len(labels)
    return crops, labels


def save_images_and_labels(crops: Sequence[Image.Image], labels: Sequence[str], split: str, data_dirname: Path):
    (data_dirname / split).mkdir(parents=True, exist_ok=True)

    with open(data_dirname / split / "_labels.json", "w") as f:
        json.dump(labels, f)
    for ind, crop in enumerate(crops):
        crop.save(data_dirname / split / f"{ind}.png")

def load_processed_crops_and_labels(split: str, data_dirname: Path):
    """Load line crops and labels for given split from processed directory."""

    crop_filenames = sorted((data_dirname / split).glob("*.png"), key=lambda filename: int(Path(filename).stem))
    crops = [util.read_image_pil(filename, grayscale=True) for filename in crop_filenames]

    with open(data_dirname / split / "_labels.json") as file:
        labels = json.load(file)
    assert len(crops) == len(labels)
    return crops, labels

def join_line_crops_form_paragraph(line_crops: Sequence[Image.Image]) -> Image.Image:
    """Horizontally stack line crops and return a single image forming the paragraph."""
    crop_shapes = np.array([_.size[::-1] for _ in line_crops])
    para_height = crop_shapes[:, 0].sum()
    para_width = crop_shapes[:, 1].max()

    para_image = Image.new(mode="L", size=(para_width, para_height), color=0)
    current_height = 0
    for line_crop in line_crops:
        para_image.paste(line_crop, box=(0, current_height))
        current_height += line_crop.height
    return para_image


def save_argument_data_as_tensors(argument_data: Sequence[Tuple[torch.Tensor, torch.Tensor]], data_dirname: Path):
    """Save argument dataset image tensors and target tensors as .pt files."""
    save_dir = data_dirname / "argument_data"
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    image_tensors = []
    target_tensors = []

    for image_tensor, target_tensor in argument_data:
        image_tensors.append(image_tensor.cpu())  # Ensure CPU storage
        target_tensors.append(target_tensor.cpu())

    # Save all tensors in a single .pt file
    torch.save(image_tensors, save_dir /"images.pt")
    torch.save(target_tensors, save_dir /"labels.pt")




def load_argument_data_as_tensors(data_dirname) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Load argument dataset image tensors and target tensors from .pt files."""
    data_dirname = Path(data_dirname)  # Convert to Path if passed as str
    save_dir = data_dirname / "argument_data"

    images = torch.load(save_dir / "images.pt")  # Load image tensors
    labels = torch.load(save_dir / "labels.pt")  # Load label tensors
    
    return images, labels



class ArgumentParagraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        line_crops: List[Image.Image],
        line_labels: List[str],
        dataset_len: int,
        inverse_mapping: dict,
        input_dims: Tuple[int, ...],
        output_dims: Tuple[int, ...],
        transform: Callable = None,
    ) -> None:
        super().__init__()
        self.line_crops = line_crops
        self.transform = transform
        self.line_labels = line_labels
        assert len(self.line_crops) == len(self.line_labels)

        self.ids = list(range(len(self.line_labels)))
        self.dataset_len = dataset_len
        self.inverse_mapping = inverse_mapping
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.min_num_lines, self.max_num_lines = 1, 15

        self.seed_set = False
        self.NEW_LINE_TOKEN = metadata_iam_paragraphs.NEW_LINE_TOKEN


    def generate_argument_paragraphs(self):
        count=0
        generated_data = []  # Store generated (image, label) pairs
        length = self.output_dims[0]

        ## original list of the crops id
        ids = list(range(len(self.line_labels)))

        while count < self.dataset_len:
            num_lines = random.randint(self.min_num_lines, min(self.max_num_lines, len(ids)))
            indices = random.sample(ids, k=num_lines)
            ## Picking random crops and stick them together as indices is randomly chosen
            datum = join_line_crops_form_paragraph([self.line_crops[i] for i in indices])
            label = (self.NEW_LINE_TOKEN).join([self.line_labels[i] for i in indices])


            if (
                (len(label) > self.output_dims[0] - 2)
               ):
                continue

            if self.transform is not None:
               datum = self.transform(datum)
            
            count += 1   
            target = convert_strings_to_labels(strings=[label], mapping=self.inverse_mapping, length=length)[0]
            generated_data.append((datum, target))

        return generated_data
    




    