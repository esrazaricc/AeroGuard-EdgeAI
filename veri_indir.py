import os
import cv2
import numpy as np

def verileri_hazirla():
    print("🛰️ AeroGuard Gerçekçi Çoklu Spektrum Veri Seti Üretici...")
    
    os.makedirs("data/rgb", exist_ok=True)
    os.makedirs("data/thermal", exist_ok=True)
    
    for i in range(1, 11):
        # 1. Gerçekçi RGB Sahnesi Üretimi (Tesis Arka Planı ve Nesneler)
        img_rgb = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Yapay bir zemin gradyanı (Piksel çeşitliliği için)
        for y in range(640):
            img_rgb[y, :] = [int(20 + y*0.1), int(30 + y*0.05), int(40 + y*0.02)]
            
        # Sahneye pist çizgileri veya hangar silüetleri ekleyelim (Kenar tespiti için)
        cv2.rectangle(img_rgb, (100, 200), (300, 500), (80, 85, 90), -1) 
        cv2.line(img_rgb, (0, 550), (640, 550), (200, 200, 200), 5) 
        
        # Şüpheli Araç/İnsan Silüeti (RGB Görünümü)
        cv2.circle(img_rgb, (400, 450), 30, (50, 50, 150), -1)
        
        cv2.imwrite(f"data/rgb/sahne_{i:03d}.jpg", img_rgb)
        
        # 2. Gerçekçi Termal Sahnesi Üretimi (Isı İmzası ve Anomali)
        img_thermal = np.zeros((640, 640), dtype=np.uint8)
        
        # Hafif ortam sıcaklığı gradyanı
        for y in range(640):
            img_thermal[y, :] = int(15 + y*0.05)
            
        # Hangarın termal yansıması
        cv2.rectangle(img_thermal, (100, 200), (300, 500), 40, -1)
        
        # ANOMALİ: RGB'de duran nesnenin termal odakta aşırı sıcak (parlak) görünmesi
        # i çiftse büyük anomali, tekse küçük anomali yapalım ki model farklı kayıplar öğrensin
        isi_degeri = 240 if i % 2 == 0 else 70 
        cv2.circle(img_thermal, (400, 450), 30, isi_degeri, -1)
        
        # Ufak bir ısınma gradyanı (Isı haritası gerçekçi dursun diye)
        cv2.GaussianBlur(img_thermal, (25, 25), 0, dst=img_thermal)
        
        cv2.imwrite(f"data/thermal/sahne_{i:03d}.jpg", img_thermal)
        
    print("✅ Yeni gerçekçi veri çiftleri başarıyla 'data/' klasörüne yazıldı!")

if __name__ == "__main__":
    verileri_hazirla()