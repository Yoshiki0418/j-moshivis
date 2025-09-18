from pathlib import Path
from typing import List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CocoLikeDataset(Dataset):
    def __init__(self, root_dir: Path, image_size: int = 256, return_pil: bool = True):
        self.root_dir = Path(root_dir)
        self.image_paths = sorted([
            p for p in self.root_dir.iterdir() if p.suffix.lower() in {'.jpg', '.png'}
        ])
        self.return_pil = return_pil
        self.image_size = image_size
        # if not returning PIL, prepare tensor transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert('RGB')
        if self.return_pil:
            return img
        return self.transform(img)


def get_dataloader(dataset: CocoLikeDataset, batch_size: int = 4) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
