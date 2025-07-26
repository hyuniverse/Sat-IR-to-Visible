# src/train.py

import time
import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import GeneratorUNet, Discriminator

# --- 데이터셋 클래스 (PIX2PIX 노트북에서 가져옴) ---
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms_
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.jpg"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))  # 가시 영상
        img_B = img.crop((w / 2, 0, w, h))  # 적외선 영상

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

# --- 가중치 초기화 함수 ---
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# --- 학습을 시작하는 메인 함수 ---
def start_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for training.")

    # 모델, 손실함수, 옵티마이저 정의
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)
    
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    criterion_GAN = nn.MSELoss().to(device)
    criterion_pixelwise = nn.L1Loss().to(device)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 데이터로더 설정
    transforms_ = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataloader = DataLoader(
        ImageDataset(args.dataset_path, transforms_=transforms_),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        ImageDataset(args.dataset_path, transforms_=transforms_, mode="val"),
        batch_size=10,
        shuffle=True
    )
    
    print(f"Start training for {args.epochs} epochs...")
    start_time = time.time()
    
    # --- 학습 루프 ---
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_dataloader):
            real_A = batch["A"].to(device) # 실제 가시광선
            real_B = batch["B"].to(device) # 실제 적외선

            valid = torch.ones(real_A.size(0), 1, 16, 16, device=device)
            fake = torch.zeros(real_A.size(0), 1, 16, 16, device=device)

            # --- 생성자(Generator) 학습 ---
            optimizer_G.zero_grad()
            fake_A = generator(real_B) # 적외선 -> 가시광선 변환
            loss_GAN = criterion_GAN(discriminator(fake_A, real_B), valid)
            loss_pixel = criterion_pixelwise(fake_A, real_A)
            loss_G = loss_GAN + args.lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            # --- 판별자(Discriminator) 학습 ---
            optimizer_D.zero_grad()
            loss_real = criterion_GAN(discriminator(real_A, real_B), valid)
            loss_fake = criterion_GAN(discriminator(fake_A.detach(), real_B), fake)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()
        
        # 로그 출력
        print(f"[Epoch {epoch+1}/{args.epochs}] [D loss: {loss_D.item():.6f}] [G loss: {loss_G.item():.6f}, pixel: {loss_pixel.item():.6f}, adv: {loss_GAN.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]")

    # 모델 저장
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(generator.state_dict(), args.model_path)
    print(f"Training finished. Model saved to {args.model_path}")
    