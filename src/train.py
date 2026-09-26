import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

ana_dizin = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ana_dizin, "src"))

from main import DualInputFusionEncoder
from decode import FusionDecoder

class SSIMLoss(nn.Module):
    def __init__(self, pencere_boyutu=11, kanal=3):
        super(SSIMLoss, self).__init__()
        self.pencere_boyutu = pencere_boyutu
        self.kanal = kanal
        self.pencereler = self.pencere_olustur(pencere_boyutu, kanal)

    def gaussian(self, pencere_boyutu, sigma):
        gauss = torch.Tensor([np.exp(-(x - pencere_boyutu // 2) ** 2 / (2 * sigma ** 2)) for x in range(pencere_boyutu)])
        return gauss / gauss.sum()

    def pencere_olustur(self, pencere_boyutu, kanal):
        _1D_pencere = self.gaussian(pencere_boyutu, 1.5).unsqueeze(1)
        _2D_pencere = torch.matmul(_1D_pencere, _1D_pencere.t()).float().unsqueeze(0).unsqueeze(0)
        pencereler = _2D_pencere.expand(kanal, 1, pencere_boyutu, pencere_boyutu).contiguous()
        return pencereler

    def forward(self, img1, img2):
        cihaz = img1.device
        self.pencereler = self.pencereler.to(cihaz)
        
        mu1 = F.conv2d(img1, self.pencereler, padding=self.pencere_boyutu // 2, groups=self.kanal)
        mu2 = F.conv2d(img2, self.pencereler, padding=self.pencere_boyutu // 2, groups=self.kanal)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.pencereler, padding=self.pencere_boyutu // 2, groups=self.kanal) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.pencereler, padding=self.pencere_boyutu // 2, groups=self.kanal) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.pencereler, padding=self.pencere_boyutu // 2, groups=self.kanal) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_haritasi = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_haritasi.mean()

class AeroGuardDataset(Dataset):
    def __init__(self, imge_klasoru, hedef_boyut=(640, 640)):
        self.imge_klasoru = imge_klasoru
        self.hedef_boyut = hedef_boyut
        self.dosyalar = [os.path.join(imge_klasoru, f) for f in os.listdir(imge_klasoru) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        self.rgb_transform = transforms.Compose([
            transforms.Resize(hedef_boyut),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dosyalar)

    def __getitem__(self, idx):
        yol = self.dosyalar[idx]
        pil_resim = Image.open(yol).convert("RGB")
        
        t_rgb = self.rgb_transform(pil_resim)
        
        np_resim = np.array(pil_resim)
        gri = cv2.cvtColor(np_resim, cv2.COLOR_RGB2GRAY)
        gri_resized = cv2.resize(gri, self.hedef_boyut)
        
        sobelx = cv2.Sobel(gri_resized, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gri_resized, cv2.CV_64F, 0, 1, ksize=5)
        gradyan = np.sqrt(sobelx**2 + sobely**2)
        gradyan = cv2.normalize(gradyan, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        sahte_termal = cv2.addWeighted(gri_resized, 0.6, gradyan, 0.4, 0)
        sahte_termal = cv2.GaussianBlur(sahte_termal, (15, 15), 0)
        
        t_termal = torch.from_numpy(sahte_termal).float().unsqueeze(0) / 255.0
        
        return t_rgb, t_termal

def egitimi_baslat(veri_klasoru, epok_sayisi=20, batch_boyutu=8, ogrenme_orani=1e-4):
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Eğitim Başlatılıyor! Aktif Cihaz: {cihaz.type.upper()}")
    
    weights_klasoru = os.path.join(ana_dizin, "weights")
    os.makedirs(weights_klasoru, exist_ok=True)
    
    veri_seti = AeroGuardDataset(veri_klasoru)
    if len(veri_seti) == 0:
        print(f"❌ HATA: '{veri_klasoru}' klasöründe geçerli resim bulunamadı!")
        return
        
    veri_yukleyici = DataLoader(veri_seti, batch_size=batch_boyutu, shuffle=True, num_workers=2)
    
    encoder = DualInputFusionEncoder(latent_dim=128).to(cihaz)
    decoder = FusionDecoder(latent_dim=128).to(cihaz)
    
    parametreler = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parametreler, lr=ogrenme_orani)
    
    mse_loss_fn = nn.MSELoss()
    ssim_loss_fn = SSIMLoss(kanal=3)
    
    encoder.train()
    decoder.train()
    
    for epoch in range(1, epok_sayisi + 1):
        toplam_kayip = 0.0
        toplam_mse = 0.0
        toplam_ssim = 0.0
        
        for adim, (rgb, termal) in enumerate(veri_yukleyici, 1):
            rgb = rgb.to(cihaz)
            termal = termal.to(cihaz)
            
            girdi = torch.cat((rgb, termal), dim=1)
            
            optimizer.zero_grad()
            gizli_kod = encoder(girdi)
            rekonstriksiyon = decoder(gizli_kod)
            
            # Decoder 4 kanal üretir: RGB + yardımcı termal. Boyutlar bilinçli olarak eşleştirilir.
            hedef = torch.cat((rgb, termal), dim=1)
            loss_mse = mse_loss_fn(rekonstriksiyon, hedef)
            # SSIM yalnızca görüntülenebilir RGB kanallarında hesaplanır.
            loss_ssim = ssim_loss_fn(rekonstriksiyon[:, :3], rgb)
            loss = loss_mse + 0.5 * loss_ssim
            
            loss.backward()
            optimizer.step()
            
            toplam_kayip += loss.item()
            toplam_mse += loss_mse.item()
            toplam_ssim += loss_ssim.item()
            
        ortalama_kayip = toplam_kayip / len(veri_yukleyici)
        ortalama_mse = toplam_mse / len(veri_yukleyici)
        ortalama_ssim = toplam_ssim / len(veri_yukleyici)
        
        print(f"Epoch [{epoch:02d}/{epok_sayisi:02d}] -> Toplam Kayıp: {ortalama_kayip:.5f} | MSE: {ortalama_mse:.5f} | SSIM Kaybı: {ortalama_ssim:.5f}")
        
        if epoch % 5 == 0:
            torch.save(encoder.state_dict(), os.path.join(weights_klasoru, f"encoder_epoch_{epoch}.pth"))
            torch.save(decoder.state_dict(), os.path.join(weights_klasoru, f"decoder_epoch_{epoch}.pth"))
            
    torch.save(encoder.state_dict(), os.path.join(weights_klasoru, "encoder_ssim.pth"))
    torch.save(decoder.state_dict(), os.path.join(weights_klasoru, "decoder_ssim.pth"))
    print("🎯 Eğitim Tamamlandı! SSIM tabanlı en başarılı ağırlıklar 'weights/' altına başarıyla kaydedildi.")

if __name__ == "__main__":
    egitim_veri_yolu = os.path.join(ana_dizin, "data", "train_images")
    os.makedirs(egitim_veri_yolu, exist_ok=True)
    egitimi_baslat(veri_klasoru=egitim_veri_yolu, epok_sayisi=20, batch_boyutu=4, ogrenme_orani=1e-4)