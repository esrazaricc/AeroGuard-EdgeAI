import torch
import torch.nn as nn

class FusionDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(FusionDecoder, self).__init__()
        
        self.fc_unpack = nn.Linear(latent_dim, 32 * 20 * 20)
        
        self.decoder_layers = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, latent_vector):
        x = self.fc_unpack(latent_vector)
        x = x.view(-1, 32, 20, 20)
        reconstructed_scene = self.decoder_layers(x)
        return reconstructed_scene

if __name__ == "__main__":
    print("--- Decoder Katmani Testi ---")
    
    mock_latent = torch.randn(2, 128)
    decoder = FusionDecoder(latent_dim=128)
    
    with torch.no_grad():
        reconstructed = decoder(mock_latent)
        
    print(f"Giris boyutu: {mock_latent.shape}")
    print(f"Cikti boyutu: {reconstructed.shape}")
    print("Test basarili, decoder sorunsuz calisiyor.")