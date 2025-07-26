# inference.py
import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from model import GeneratorUNet

class SatelliteDataset(Dataset):
    def __init__(self, root, transforms_):
        self.transform = transforms_
        self.files = sorted(glob.glob(os.path.join(root, "*.png")))

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, os.path.basename(img_path)

    def __len__(self):
        return len(self.files)

def translate_images(input_dir, output_dir, model_path):
    """
    지정된 폴더의 적외선 이미지를 가시광선 이미지로 변환합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 불러오기
    generator = GeneratorUNet().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # 데이터셋 및 데이터로더 준비
    transforms_ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = SatelliteDataset(input_dir, transforms_=transforms_)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 변환 및 저장
    for i, (real_ir, filename) in enumerate(dataloader):
        real_ir = real_ir.to(device)
        with torch.no_grad():
            fake_vi = generator(real_ir)
        
        # 생성된 가시광선 이미지 저장
        save_image(fake_vi.data, os.path.join(output_dir, filename[0]), normalize=True)

        print(f"[{i+1}/{len(dataloader)}] '{filename[0]}' 변환 완료")