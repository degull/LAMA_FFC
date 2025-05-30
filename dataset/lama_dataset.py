# dataset/lama_dataset.py
# celeb
""" 
import os
import random
import glob
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LaMaDataset(Dataset):
    def __init__(self, image_dir, mask_type="irregular", image_size=256, phase="train"):

        self.image_dir = image_dir
        # ✅ 하위 폴더까지 모두 포함한 이미지 경로 수집
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True) +
                                  glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True))
        self.mask_type = mask_type
        self.image_size = image_size
        self.phase = phase
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        mask = self.generate_mask(self.image_size, self.image_size)
        masked_image = image * (1 - mask)  # x ⊙ m

        input_tensor = torch.cat([masked_image, mask], dim=0)  # 4 channels
        return {
            "image": image,         # Ground truth
            "mask": mask,           # Binary mask (1 = hole)
            "input": input_tensor,  # 4ch input: [masked_image + mask]
            "path": self.image_paths[idx]
        }

    def generate_mask(self, h, w):
        if self.mask_type == "irregular":
            return self._generate_irregular_mask(h, w)
        elif self.mask_type == "box":
            return self._generate_box_mask(h, w)
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")

    def _generate_irregular_mask(self, h, w):
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        n_strokes = random.randint(1, 5)
        for _ in range(n_strokes):
            n_vertex = random.randint(4, 8)
            points = []
            for _ in range(n_vertex):
                x = random.randint(0, w)
                y = random.randint(0, h)
                points.append((x, y))
            width = random.randint(10, 40)
            draw.line(points, fill=255, width=width)

        mask = transforms.ToTensor()(mask)  # [1, H, W]
        mask = (mask > 0.5).float()
        return mask

    def _generate_box_mask(self, h, w):
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for _ in range(random.randint(1, 5)):
            x1 = random.randint(0, w - 32)
            y1 = random.randint(0, h - 32)
            box_w = random.randint(32, w // 2)
            box_h = random.randint(32, h // 2)
            draw.rectangle([x1, y1, x1 + box_w, y1 + box_h], fill=255)

        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()
        return mask
 """

# standard 365
import os
import random
import glob
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LaMaDataset(Dataset):
    def __init__(self, image_dir, mask_type="irregular", image_size=256, phase="train"):
        """
        Args:
            image_dir (str): 이미지 폴더 (Places or CelebA-HQ root)
            mask_type (str): "irregular", "box", or "segmentation"
            image_size (int): 입력 이미지 크기 (256 또는 512 등)
            phase (str): "train", "val", or "test"
        """
        self.image_dir = image_dir

        # ✅ 평평한 구조에서 모든 .jpg, .png 이미지 수집
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                                  glob.glob(os.path.join(image_dir, "*.png")))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        print(f"✅ LaMaDataset: {len(self.image_paths)} images loaded from {image_dir}")

        self.mask_type = mask_type
        self.image_size = image_size
        self.phase = phase
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        mask = self.generate_mask(self.image_size, self.image_size)
        masked_image = image * (1 - mask)

        input_tensor = torch.cat([masked_image, mask], dim=0)  # 4 channels
        return {
            "image": image,
            "mask": mask,
            "input": input_tensor,
            "path": self.image_paths[idx]
        }

    def generate_mask(self, h, w):
        if self.mask_type == "irregular":
            return self._generate_irregular_mask(h, w)
        elif self.mask_type == "box":
            return self._generate_box_mask(h, w)
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")

    def _generate_irregular_mask(self, h, w):
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        n_strokes = random.randint(1, 5)
        for _ in range(n_strokes):
            n_vertex = random.randint(4, 8)
            points = [(random.randint(0, w), random.randint(0, h)) for _ in range(n_vertex)]
            width = random.randint(10, 40)
            draw.line(points, fill=255, width=width)

        mask = transforms.ToTensor()(mask)
        return (mask > 0.5).float()

    def _generate_box_mask(self, h, w):
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for _ in range(random.randint(1, 5)):
            x1 = random.randint(0, w - 32)
            y1 = random.randint(0, h - 32)
            box_w = random.randint(32, w // 2)
            box_h = random.randint(32, h // 2)
            draw.rectangle([x1, y1, x1 + box_w, y1 + box_h], fill=255)

        mask = transforms.ToTensor()(mask)
        return (mask > 0.5).float()
