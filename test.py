# test.py
import os
import torch
from torchvision.utils import save_image
from models.generator import LaMaGenerator
from dataset.lama_dataset import LaMaDataset
from torch.utils.data import DataLoader

def test():
    image_dir = "data/val"
    save_dir = "results"
    checkpoint = "checkpoints/lama_epoch20.pth"
    image_size = 256
    mask_type = "irregular"

    os.makedirs(save_dir, exist_ok=True)

    dataset = LaMaDataset(image_dir, mask_type=mask_type, image_size=image_size, phase="val")
    dataloader = DataLoader(dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LaMaGenerator(in_channels=4).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    for i, batch in enumerate(dataloader):
        input_tensor = batch["input"].to(device)
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        with torch.no_grad():
            output = model(input_tensor)

        save_image(output, os.path.join(save_dir, f"{i}_output.png"))
        save_image(image, os.path.join(save_dir, f"{i}_gt.png"))
        save_image(mask, os.path.join(save_dir, f"{i}_mask.png"))

if __name__ == "__main__":
    test()
