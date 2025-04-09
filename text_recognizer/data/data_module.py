from text_recognizer.data.create_save_argument_dataset import load_argument_data_as_tensors
from torch.utils.data import Dataset, DataLoader, random_split
from text_recognizer.data.base_data_module import BaseDataModule
from text_recognizer.stems.paragraph import ParagraphStem
from text_recognizer.data.create_save_argument_dataset import (load_processed_crops_and_labels,
                                                               extract_images_and_labels,
                                                               ArgumentParagraphDataset,
                                                               input_dims,
                                                               output_dims,
                                                               inverse_mapping)




class CustomDataset(Dataset):
    def __init__(self, data_dir, dataset_len):
        crops, labels=load_processed_crops_and_labels(split="train", data_dirname=data_dir)
        # Create dataset
        argument_dataset = ArgumentParagraphDataset(
            line_crops=crops,
            line_labels=labels,
            dataset_len=dataset_len,
            inverse_mapping=inverse_mapping,
            input_dims=input_dims,
            output_dims=output_dims,
            transform=ParagraphStem(augment=False),
        )

        # Generate training data
        argument_data = argument_dataset.generate_argument_paragraphs()
        self.images, self.targets = extract_images_and_labels(argument_data)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


class ArgumentDataModule(BaseDataModule):
    def __init__(self, data_dir, dataset_len, batch_size, val_split=0.2, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = CustomDataset(self.data_dir, self.dataset_len)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
