import os
import torch
from main import DualInputFusionEncoder

def onnx_donustur():
    print("--- AeroGuard Fusion Model Optimization Pipeline ---")
    
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_boyut = 128
    
    # Eğittiğimiz gerçek encoder modelini ayağa kaldırıyoruz
    model = DualInputFusionEncoder(latent_dim=latent_boyut)
    
    # Eğer önceden train.py çalıştırıp ağırlık kaydettiysek onu yüklüyoruz.
    # Dosya yoksa bile kodun çökmemesi için kontrol ekledik.
    proje_dizini = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agirlik_yolu = os.path.join(proje_dizini, "weights", "encoder_ssim.pth")
    if os.path.exists(agirlik_yolu):
        model.load_state_dict(torch.load(agirlik_yolu, map_location=cihaz))
        print("Egitilmis model agirliklari yuklendi.")
    else:
        print("Egitilmis agirlik bulunamadi  model rastgele agirliklarla donustürülüyor.")
        
    model.to(cihaz)
    model.eval()
    sahte_girdi = torch.randn(1, 4, 640, 640, device=cihaz)
    
    weights_dizini = os.path.join(proje_dizini, "weights")
    os.makedirs(weights_dizini, exist_ok=True)
    cikti_onnx_yolu = os.path.join(weights_dizini, "fusion_encoder.onnx")

    print("ONNX formatina donusum baslatiliyor...")
    
    torch.onnx.export(
        model,
        sahte_girdi,
        cikti_onnx_yolu,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_rgb_t'],
        output_names=['output_latent'],
        dynamic_axes={'input_rgb_t': {0: 'batch_size'}, 'output_latent': {0: 'batch_size'}}
    )
    
    print(f"Islem basarili, ONNX modeli kaydedildi: {cikti_onnx_yolu}")

if __name__ == "__main__":
    onnx_donustur()

