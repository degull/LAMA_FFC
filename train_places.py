# train.py
# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.generator import LaMaGenerator
from models.discriminator_multi import MultiScaleDiscriminator
from loss.losses import LaMaLoss
from dataset.lama_dataset import LaMaDataset

def main():
    # ✅ 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = "E:/이미지개선/FFT/data/Places365/sampled_train"
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 데이터셋 로딩
    train_dataset = LaMaDataset(image_dir=image_dir, image_size=256, mask_type="irregular")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)  # ⚠️ Windows는 num_workers=0 안정적

    # ✅ 모델 초기화
    generator = LaMaGenerator(in_channels=4).to(device)
    discriminator = MultiScaleDiscriminator().to(device)
    loss_fn = LaMaLoss(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # ✅ 훈련 루프
    for epoch in range(50):
        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            masked_images = images * (1 - masks)
            inputs = torch.cat([masked_images, masks], dim=1)  # 4채널 입력

            # 🔹 Generator
            fake_images = generator(inputs)
            pred_fake = discriminator(fake_images)
            g_loss_dict = loss_fn(fake_images, images, pred_fake)
            loss_g = g_loss_dict["loss_generator_total"]

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            # 🔹 Discriminator
            pred_fake = discriminator(fake_images.detach())
            pred_real = discriminator(images)
            d_loss = loss_fn(fake_images.detach(), images, pred_fake, pred_real)["loss_discriminator"]

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch}][Iter {i}] G: {loss_g.item():.4f}, D: {d_loss.item():.4f}")

        # ✅ Epoch마다 결과 저장
        save_image(fake_images[:4], os.path.join(save_dir, f"epoch_{epoch}_fake.png"))
        save_image(images[:4], os.path.join(save_dir, f"epoch_{epoch}_real.png"))
        torch.save(generator.state_dict(), os.path.join(save_dir, f"lama_generator_epoch{epoch}.pth"))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # ✅ Windows 호환성 확보
    main()
