# train.py
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.generator import LaMaGenerator
from models.discriminator_multi import MultiScaleDiscriminator
from dataset.lama_dataset import LaMaDataset
from loss.losses import LaMaLoss
from utils.utils import save_sample_images
from utils.detectron_mask_generator import setup_detectron2

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional: Detectron2 predictor for segmentation masks
    predictor = setup_detectron2() if cfg["data"]["mask_type"] == "segmentation" else None

    # Dataset & Dataloader
    train_set = LaMaDataset(
        image_dir=cfg["data"]["train_dir"],
        mask_type=cfg["data"]["mask_type"],
        image_size=cfg["data"]["image_size"],
        phase="train",
        predictor=predictor
    )
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=4)

    # Models
    generator = LaMaGenerator(in_channels=4).to(device)
    discriminator = MultiScaleDiscriminator(num_scales=3).to(device)

    # Loss & Optimizer
    loss_fn = LaMaLoss(
        perceptual_weight=cfg["loss"]["perceptual"],
        adv_weight=cfg["loss"]["adv"],
        fm_weight=cfg["loss"]["feature_matching"]
    )
    opt_G = optim.Adam(generator.parameters(), lr=cfg["train"]["lr"])
    opt_D = optim.Adam(discriminator.parameters(), lr=cfg["train"]["lr"])

    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        generator.train()
        discriminator.train()

        for i, batch in enumerate(train_loader):
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            input_tensor = batch["input"].to(device)

            # --- Generator forward ---
            fake = generator(input_tensor)

            # --- Discriminator forward ---
            pred_real = discriminator(img)
            pred_fake = discriminator(fake.detach())

            # --- D loss (Multi-scale hinge loss) ---
            loss_D = sum([
                torch.mean(nn.ReLU()(1.0 - real)) + torch.mean(nn.ReLU()(1.0 + fake))
                for real, fake in zip(pred_real, pred_fake)
            ]) / len(pred_real)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- G loss ---
            pred_fake_G = discriminator(fake)
            losses = loss_fn(fake, img, pred_fake_G)
            opt_G.zero_grad()
            losses["loss_total"].backward()
            opt_G.step()

            if i % cfg["train"]["log_interval"] == 0:
                print(f"[Epoch {epoch}/{cfg['train']['epochs']}][{i}/{len(train_loader)}] "
                      f"Loss_G: {losses['loss_total']:.4f}, Loss_D: {loss_D:.4f}")

        # Save model
        torch.save(generator.state_dict(), os.path.join(cfg["train"]["save_dir"], f"lama_epoch{epoch}.pth"))

        # Save sample output
        if epoch % cfg["train"]["sample_interval"] == 0:
            save_sample_images(fake, img, mask, cfg["train"]["save_dir"], epoch)

if __name__ == "__main__":
    main()
