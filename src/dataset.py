import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class MultispectralRiskDataset(Dataset):
    def __init__(self, rgb_dizin="data/rgb", termal_dizin="data/thermal"):
        self.rgb_dizin = rgb_dizin
        self.termal_dizin = termal_dizin
        
        if os.path.exists(rgb_dizin):
            self.rgb_listesi = sorted([f for f in os.listdir(rgb_dizin) if not f.startswith('.')])
        else:
            self.rgb_listesi = []
            
        if os.path.exists(termal_dizin):
            self.termal_listesi = sorted([f for f in os.listdir(termal_dizin) if not f.startswith('.')])
        else:
            self.termal_listesi = []

    def __len__(self):
        return len(self.rgb_listesi)

    def __getitem__(self, indis):
        yol_rgb = os.path.join(self.rgb_dizin, self.rgb_listesi[indis])
        ham_rgb = np.fromfile(yol_rgb, dtype=np.uint8)
        gorsel_rgb = cv2.imdecode(ham_rgb, cv2.IMREAD_COLOR)
        
        if gorsel_rgb is None:
            raise FileNotFoundError(f"Yukleme hatasi: {yol_rgb}")
            
        gorsel_rgb = cv2.cvtColor(gorsel_rgb, cv2.COLOR_BGR2RGB)
        gorsel_rgb = cv2.resize(gorsel_rgb, (640, 640))
        tensor_rgb = torch.from_numpy(gorsel_rgb).permute(2, 0, 1).float() / 255.0
        
        yol_termal = os.path.join(self.termal_dizin, self.termal_listesi[indis])
        ham_termal = np.fromfile(yol_termal, dtype=np.uint8)
        gorsel_termal = cv2.imdecode(ham_termal, cv2.IMREAD_GRAYSCALE)
        
        if gorsel_termal is None:
            raise FileNotFoundError(f"Yukleme hatasi: {yol_termal}")
            
        gorsel_termal = cv2.resize(gorsel_termal, (640, 640))
        tensor_termal = torch.from_numpy(gorsel_termal).unsqueeze(0).float() / 255.0
        
        return tensor_rgb, tensor_termal