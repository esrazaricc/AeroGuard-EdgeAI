import torch
import torch.nn as nn

class DualInputFusionEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(DualInputFusionEncoder, self).__init__()
        
        self.rgb_kolu = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.termal_kolu = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.birlesme_katmani = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.duzlestir = nn.Flatten()
        self.fc_katmani = nn.Linear(32 * 20 * 20, latent_dim)

    def forward(self, veri):
        rgb_kanallari = veri[:, :3, :, :]
        termal_kanali = veri[:, 3:, :, :]
        
        rgb_ozellik = self.rgb_kolu(rgb_kanallari)
        termal_ozellik = self.termal_kolu(termal_kanali)
        
        bilesik_vektor = torch.cat((rgb_ozellik, termal_ozellik), dim=1)
        ara_katman = self.birlesme_katmani(bilesik_vektor)
        cikti = self.fc_katmani(self.duzlestir(ara_katman))
        
        return cikti

if __name__ == "__main__":
    print("--- Encoder Katmani Testi ---")
    
    sahte_giris = torch.randn(2, 4, 640, 640)
    model = DualInputFusionEncoder(latent_dim=128)
    
    with torch.no_grad():
        cikti_vektörü = model(sahte_giris)
        
    print(f"Giris boyutu: {sahte_giris.shape}")
    print(f"Cikti boyutu: {cikti_vektörü.shape}")
    print("Test basarili, encoder sorunsuz calisiyor.")