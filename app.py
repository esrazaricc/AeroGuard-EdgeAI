import os
import sys
import cv2
import torch
import numpy as np
import streamlit as st
from datetime import datetime
from ultralytics import YOLO

# Proje dizin ayarları
ana_dizin = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ana_dizin, "src"))

from main import DualInputFusionEncoder
from decode import FusionDecoder

# Streamlit sayfa yapılandırması
st.set_page_config(page_title="AeroGuard Sınıflandırmalı Takip", layout="wide")

def telegram_bildirim_gonder(mesaj):
    """Telegram bildirimi yalnızca ortam değişkenleri tanımlıysa gönderilir."""
    import requests
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    veri = {"chat_id": chat_id, "text": mesaj, "parse_mode": "HTML"}
    try:
        yanit = requests.post(url, json=veri, timeout=3)
        yanit.raise_for_status()
        return True
    except requests.RequestException:
        return False

@st.cache_resource
def model_yukle():
    donanim = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model = DualInputFusionEncoder(latent_dim=128).to(donanim)
    decoder_model = FusionDecoder(latent_dim=128).to(donanim)
    yol_encoder = os.path.join(ana_dizin, "weights", "encoder_ssim.pth")
    yol_decoder = os.path.join(ana_dizin, "weights", "decoder_ssim.pth")

    if os.path.exists(yol_encoder) and os.path.exists(yol_decoder):
        encoder_model.load_state_dict(torch.load(yol_encoder, map_location=donanim))
        decoder_model.load_state_dict(torch.load(yol_decoder, map_location=donanim))

    encoder_model.eval()
    decoder_model.eval()

    yolo_yolu = os.path.join(ana_dizin, "yolov8m.pt")
    if not os.path.exists(yolo_yolu):
        raise FileNotFoundError(f"YOLO ağırlığı bulunamadı: {yolo_yolu}")
    yolo_nesne_modeli = YOLO(yolo_yolu)

    return encoder_model, decoder_model, yolo_nesne_modeli, donanim

enc, dec, yolo_detektor, cihaz = model_yukle()

if "ihlal_gunlugu" not in st.session_state:
    st.session_state["ihlal_gunlugu"] = []

if "ardisik_hareket_sayaci" not in st.session_state:
    st.session_state["ardisik_hareket_sayaci"] = 0

st.title("🛰️ AeroGuard Canlı Kamera Tehdit ve Yabancı Cisim Tespit Sistemi")

sol_kolon, sag_kolon = st.columns(2)

with sol_kolon:
    st.subheader("Kritik Bölge Kaynak Ayarları")
    kaynak_turu = st.selectbox("Giriş Kaynağı Seçin", ["Yerel Web Kamerası", "Yerel Test Videosu (MP4)", "IP Kamera / RTSP Akışı"])

    kamera_adresi = 0
    if kaynak_turu == "IP Kamera / RTSP Akışı":
        kamera_adresi = st.text_input("Kamera RTSP / HTTP Bağlantı Adresi", "rtsp://admin:password@11.22.33.44:554/stream")
    elif kaynak_turu == "Yerel Test Videosu (MP4)":
        kamera_adresi = st.text_input("Dosya Yolu", "data/test_drone.mp4")

    st.write("🛡️ **Erken Uyarı Koruma Koridoru (Yükseklik Ayarı)**")
    koruma_sınırı = st.slider("Koridor Sınır Çizgisi (Yukarıdan Aşağıya Piksel)", 50, 600, 300)

    # --- 🔧 YENİ: Hassasiyet ayarları (debug/ince ayar için) ---
    with st.expander("⚙️ Gelişmiş Tespit Hassasiyeti"):
        yolo_guven_esigi = st.slider("YOLO Genel Güven Eşiği", 0.05, 0.50, 0.15, 0.01)
        arac_min_guven = st.slider("Araç Sınıfı Minimum Güven", 0.10, 0.60, 0.35, 0.01)
        kontur_alan_esigi = st.slider("Minimum Hareket Alanı (piksel²)", 5, 200, 20, 5)
        arka_plan_ogrenme_orani = st.slider("Arka Plan Öğrenme Oranı", 0.0001, 0.05, 0.005, 0.0001, format="%.4f")
        ardisik_onay_esigi = st.slider("Gürültü Filtresi (Ardışık Kare Onayı)", 1, 10, 4, 1,
                                        help="Sadece piksel bazlı risk sinyali (nesne sınıflandırılmadan) tehdit saymadan önce kaç ardışık karede görülmesi gerektiği. Yüksek değer = daha az yanlış alarm, ama tepki biraz gecikir.")
        debug_modu = st.checkbox("🐞 Debug: Ham hareket maskesini göster", value=False)

        st.markdown("---")
        st.caption("📷 Zaman damgası, kayıt göstergesi veya logo gibi sabit köşe overlay'leri hareket olarak algılanıyorsa buradan hariç tutabilirsiniz.")
        hafiz_bolgesi_aktif = st.checkbox("Köşe / Overlay Bölgesini Hariç Tut", value=True)
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            haric_konum = st.selectbox("Bölge Konumu", ["Sol Üst", "Sağ Üst", "Sol Alt", "Sağ Alt"], index=0)
        with h_col2:
            haric_boyut = st.slider("Bölge Boyutu (piksel)", 20, 200, 90, 10)

    baslat_butonu = st.checkbox("Kritik Bölge Takibini Başlat")
    kamera_ekrani = st.empty()

with sag_kolon:
    st.subheader("Anlık Risk Analizi ve Hedef Takibi")
    gosterim_modu = st.radio("Sağ Ekran Görünüm Modu", ["Isı Haritası (Risk Haritası)", "Sensör Füzyon Çıktısı (Hibrit Mod)"], horizontal=True)
    analiz_ekrani = st.empty()
    metrik_ekrani = st.empty()
    durum_ekrani = st.empty()

if 'debug_modu' in dir() and debug_modu:
    debug_ekrani = st.empty()

st.markdown("---")
st.subheader("📜 Kritik Sistem Kararları ve İhlal Günlüğü (Log)")
log_alani = st.empty()

if baslat_butonu:
    kamera = cv2.VideoCapture(kamera_adresi)
    arka_plan_modeli = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=False)
    kare_sayaci = 0

    while kamera.isOpened():
        durum, kare = kamera.read()
        if not durum:
            if kaynak_turu == "Yerel Test Videosu (MP4)":
                kamera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                st.error("Kritik bölge akışı kesildi.")
                break

        if kaynak_turu == "Yerel Web Kamerası":
            kare = cv2.flip(kare, 1)

        orijinal_boyut_y, orijinal_boyut_x = kare.shape[0], kare.shape[1]

        kare_rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        kare_boyutlu = cv2.resize(kare_rgb, (640, 640))

        kare_sayaci += 1

        gri_kare = cv2.cvtColor(kare_boyutlu, cv2.COLOR_RGB2GRAY)
        # 🔧 DÜZELTME: Sabit/düşük öğrenme oranı -> yavaş hareket eden araç arka plana "yedirilmesin"
        hareket_maskesi = arka_plan_modeli.apply(gri_kare, learningRate=arka_plan_ogrenme_orani)

        maske_alani = np.zeros_like(hareket_maskesi)
        maske_alani[0:koruma_sınırı, :] = 255

        # 🔧 DÜZELTME: Videoya gömülü zaman damgası / kayıt göstergesi / logo gibi sabit köşe
        # overlay'leri her karede piksel değişikliği ürettiği için gerçek hareketten ayırt edilemiyordu.
        # Kullanıcının seçtiği köşeyi analiz dışına (0) alıyoruz.
        if hafiz_bolgesi_aktif:
            if haric_konum == "Sol Üst":
                maske_alani[0:haric_boyut, 0:haric_boyut] = 0
            elif haric_konum == "Sağ Üst":
                maske_alani[0:haric_boyut, 640 - haric_boyut:640] = 0
            elif haric_konum == "Sol Alt":
                maske_alani[640 - haric_boyut:640, 0:haric_boyut] = 0
            elif haric_konum == "Sağ Alt":
                maske_alani[640 - haric_boyut:640, 640 - haric_boyut:640] = 0

        hareket_maskesi = cv2.bitwise_and(hareket_maskesi, maske_alani)

        # 🔧 DÜZELTME: 5x5 -> 3x3 çekirdek. Büyük çekirdek küçük/ince siluetleri (uzaktaki araç gibi) komple siliyordu
        cekirdek = np.ones((3, 3), np.uint8)
        temiz_maske = cv2.morphologyEx(hareket_maskesi, cv2.MORPH_OPEN, cekirdek)
        # Küçük boşlukları kapatmak için hafif bir "closing" ekleyelim (parçalanmış siluetleri birleştirir)
        temiz_maske = cv2.morphologyEx(temiz_maske, cv2.MORPH_CLOSE, cekirdek, iterations=2)

        if 'debug_modu' in dir() and debug_modu:
            debug_ekrani.image(temiz_maske, caption="Ham Hareket Maskesi (temiz_maske)", width=450)

        sahte_termal = cv2.GaussianBlur(temiz_maske, (21, 21), 0)

        t_rgb = torch.from_numpy(kare_boyutlu).permute(2, 0, 1).float() / 255.0
        t_termal = torch.from_numpy(sahte_termal).unsqueeze(0).float() / 255.0

        t_rgb = t_rgb.unsqueeze(0).to(cihaz)
        t_termal = t_termal.unsqueeze(0).to(cihaz)

        with torch.no_grad():
            girdi_matrisi = torch.cat((t_rgb, t_termal), dim=1)
            gizli_kod = enc(girdi_matrisi)
            tahmin_matrisi = dec(gizli_kod)

            if gosterim_modu == "Sensör Füzyon Çıktısı (Hibrit Mod)":
                # Decoder 4 kanal (RGB + yardımcı termal) üretir; ekranda yalnızca RGB gösterilir.
                fuzyon_ciktisi = tahmin_matrisi[0, :3].permute(1, 2, 0).cpu().numpy()
                fuzyon_ciktisi = np.clip(fuzyon_ciktisi * 255.0, 0, 255).astype(np.uint8)

            if kare_sayaci < 45:
                risk_skoru = np.zeros((640, 640), dtype=np.uint8)
                yuzde_risk = 0.0
            else:
                risk_skoru = cv2.GaussianBlur(sahte_termal, (15, 15), 0)
                # 🔧 DÜZELTME: yuzde_risk artık bulanıklaştırılmış (sahte_termal/risk_skoru) piksellerden değil,
                # ham temiz_maske'den hesaplanıyor. Çift Gaussian blur (21x21 + 15x15) küçük/sabit bir
                # artefaktı (toz, lens yansıması, hatalı piksel) bile gerçek boyutunun çok üzerinde bir
                # alana yayıp sürekli düşük seviyeli risk üretebiliyordu. Ham maske kullanmak ve aynı minimum
                # alan eşiğini (kontur_alan_esigi) uygulamak bu sızıntıyı kapatıyor.
                ham_hareket_pikselleri = cv2.countNonZero(temiz_maske)
                if ham_hareket_pikselleri < kontur_alan_esigi:
                    yuzde_risk = 0.0
                else:
                    koruma_alani_pikselleri = 640 * koruma_sınırı
                    oran = (ham_hareket_pikselleri / (koruma_alani_pikselleri + 1e-8)) * 10.0
                    yuzde_risk = min(oran * 100.0, 100.0)

            insan_sayisi = 0
            arac_sayisi = 0
            drone_sayisi = 0
            hareketli_bilinmeyen_cisim = 0
            hareketli_arac_benzeri = 0

            # 🔧 DÜZELTME: Sabit 0.25 yerine kullanıcı ayarlı düşük eşik (varsayılan 0.15)
            # -> Ağaç kenarından kısmen kapanarak giren araçlar artık YOLO tarafından da yakalanabilir
            yolo_sonuclari = yolo_detektor(kare_rgb, imgsz=1280, conf=yolo_guven_esigi, iou=0.45, verbose=False)

            # YOLO'nun bulduğu alanları hafızada tutmak için liste oluşturuyoruz
            yolo_kutulari = []

            for tespit in yolo_sonuclari[0].boxes:
                sinif_id = int(tespit.cls[0])
                guven_skoru = float(tespit.conf[0])
                koordinat_orijinal = tespit.xyxy[0].cpu().numpy()
                ox1, oy1, ox2, oy2 = map(int, koordinat_orijinal)

                x1 = int(ox1 * (640.0 / orijinal_boyut_x))
                y1 = int(oy1 * (640.0 / orijinal_boyut_y))
                x2 = int(ox2 * (640.0 / orijinal_boyut_x))
                y2 = int(oy2 * (640.0 / orijinal_boyut_y))

                orta_y = int((y1 + y2) / 2)

                if orta_y < koruma_sınırı:
                    if sinif_id == 0:
                        if guven_skoru < 0.50:
                            continue
                        insan_sayisi += 1
                        etiket = f"INSAN (%{guven_skoru*100:.0f})"
                        renk = (0, 0, 255)
                    elif sinif_id in [2, 3, 5, 7]:
                        # 🔧 DÜZELTME: Düşük güvende YOLO küçük/uzak nesnelerde sınıfı karıştırabiliyor
                        # (örn. ayakta duran bir insan "araç" sınıfına kayabiliyor). Bunu iki şekilde önlüyoruz:
                        # 1) Araç sınıfı için ayrı ve daha yüksek bir minimum güven eşiği (arac_min_guven)
                        # 2) Şekil kontrolü: gerçek bir araç kutusu genelde eninden geniş/kareye yakındır,
                        #    insan silueti ise dar ve dikeydir (boy > en * 1.3 ise muhtemelen insan, atla)
                        kutu_genislik = x2 - x1
                        kutu_yukseklik = y2 - y1
                        if guven_skoru < arac_min_guven:
                            continue
                        if kutu_yukseklik > kutu_genislik * 1.3:
                            continue
                        arac_sayisi += 1
                        etiket = f"ARAC (%{guven_skoru*100:.0f})"
                        renk = (255, 165, 0)
                    elif sinif_id == 4:
                        if guven_skoru < 0.40:
                            continue
                        etiket = f"UCAK (%{guven_skoru*100:.0f})"
                        renk = (0, 255, 255)
                    elif sinif_id == 8:
                        if guven_skoru < 0.40:
                            continue
                        drone_sayisi += 1
                        etiket = f"DRONE (%{guven_skoru*100:.0f})"
                        renk = (255, 0, 0)
                    else:
                        continue

                    # YOLO kutusunu kaydedelim
                    yolo_kutulari.append((x1, y1, x2, y2))
                    cv2.rectangle(kare_boyutlu, (x1, y1), (x2, y2), renk, 2)
                    cv2.putText(kare_boyutlu, etiket, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 2)

            # --- 🚀 HAREKET BAZLI DİNAMİK YABANCI CİSİM VE ARAC TAKİBİ ---
            # Eğer YOLO aracı tamamen kaçırsa bile, hareket maskesinde bir hareket varsa o bölgeyi bulup etiketliyoruz
            konturlar, _ = cv2.findContours(temiz_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for kontur in konturlar:
                # 🔧 DÜZELTME: Eşik 80 -> kullanıcı ayarlı (varsayılan 20).
                # Ağaç kenarından uzakta/küçük görünen araçların silueti 80 pikselin altında kalıp tamamen eleniyordu
                if cv2.contourArea(kontur) > kontur_alan_esigi:
                    kx, ky, kw, kh = cv2.boundingRect(kontur)

                    # Bu hareketli bölgenin zaten YOLO tarafından etiketlenip etiketlenmediğini kontrol edelim
                    cakisma_var = False
                    for (yx1, yy1, yx2, yy2) in yolo_kutulari:
                        # Eğer hareket eden alan ile YOLO kutusu üst üste biniyorsa çakışma kabul et
                        if not (kx + kw < yx1 or kx > yx2 or ky + kh < yy1 or ky > yy2):
                            cakisma_var = True
                            break

                    # Eğer YOLO bu hareketi nesne olarak yakalayamadıysa, biz devreye giriyoruz!
                    if not cakisma_var:
                        # 🔧 DÜZELTME: Çubuk, direk, bacak gibi ince/uzun cisimler bir aracın siluetine
                        # benzemez. En-boy oranına bakarak bunları "araç" sayımından ayırıyoruz.
                        # kisa_kenar küçük ve uzun_kenar/kisa_kenar oranı yüksekse -> muhtemelen ince bir cisim (çubuk/direk/bacak)
                        kisa_kenar = min(kw, kh)
                        uzun_kenar = max(kw, kh)
                        en_boy_orani = uzun_kenar / max(kisa_kenar, 1)
                        # kh > kw*1.2: kutu genişliğinden belirgin şekilde uzun -> muhtemelen dik duran bir insan/bacak
                        insan_oranli_mi = kh > kw * 1.2
                        # 🔧 DÜZELTME: Rüzgarda sallanan ağaç dalları/yapraklar dağınık, düşük "doluluk" oranlı
                        # hareket bölgeleri üretir (kutunun sadece küçük bir kısmı gerçekten hareket eden piksel içerir).
                        # Gerçek bir araç siluetinin kutusu ise büyük oranda dolu olur. Bu yüzden doluluk düşükse
                        # (örn. ağaç/yaprak kümesi) araç sayılmaz.
                        kutu_alani = kw * kh
                        doluluk_orani = cv2.contourArea(kontur) / max(kutu_alani, 1)
                        yaprak_benzeri_mi = doluluk_orani < 0.35

                        if (kisa_kenar < 18 and en_boy_orani > 3.0) or insan_oranli_mi or yaprak_benzeri_mi:
                            # İnce/uzun, insan oranlı ya da dağınık (ağaç/yaprak) hareket -> araç değil, genel "sınıflandırılmamış hareket"
                            hareketli_bilinmeyen_cisim += 1
                            cv2.rectangle(kare_boyutlu, (kx, ky), (kx + kw, ky + kh), (128, 128, 128), 2)
                            cv2.putText(kare_boyutlu, "HAREKET (INCE CISIM)", (kx, ky - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                        else:
                            # Araç siluetine daha yakın şekil/oran/doluluk -> araç adayı olarak say
                            hareketli_arac_benzeri += 1
                            cv2.rectangle(kare_boyutlu, (kx, ky), (kx + kw, ky + kh), (255, 165, 0), 2)
                            cv2.putText(kare_boyutlu, "ARAC / HEDEF (HASSAS)", (kx, ky - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

            # 🔧 DÜZELTME: Sadece YOLO'nun gerçekten sınıflandırdığı nesneler (insan/araç/drone) anında
            # güvenilir sayılır. Hareket-bazlı "araç benzeri" tahmin (hareketli_arac_benzeri) sadece bir
            # sezgiseldir; rüzgarda sallanan ağaçlar gibi şeyler de bu şekle uyabildiği için araç henüz
            # kadraja girmeden yanlışlıkla anında alarm üretebiliyordu. Artık bu da gürültü filtresinden geçiyor.
            guvenilir_hedef_sayisi = insan_sayisi + arac_sayisi + drone_sayisi
            hareket_adayi_sayisi = hareketli_arac_benzeri + hareketli_bilinmeyen_cisim
            aktif_hedef_sayisi = guvenilir_hedef_sayisi + hareket_adayi_sayisi

            cv2.line(kare_boyutlu, (0, koruma_sınırı), (640, koruma_sınırı), (0, 255, 0) if aktif_hedef_sayisi == 0 else (255, 0, 0), 2)
            kamera_ekrani.image(kare_boyutlu, channels="RGB", width=450)

            if gosterim_modu == "Isı Haritası (Risk Haritası)":
                renkli_harita = cv2.applyColorMap(risk_skoru, cv2.COLORMAP_JET)
                renkli_harita = cv2.cvtColor(renkli_harita, cv2.COLOR_BGR2RGB)
                analiz_ekrani.image(renkli_harita, width=450)
            else:
                analiz_ekrani.image(fuzyon_ciktisi, channels="RGB", width=450)

            metrik_ekrani.metric(label="Maksimum Tehdit Yoğunluğu", value=f"%{yuzde_risk:.2f}", delta=f"{aktif_hedef_sayisi} Sınıflandırılmış Hedef")

            # 🔧 DÜZELTME: Sadece güvenilir (YOLO sınıflandırmalı) nesneler anında tetikler.
            # Hareket adayları (araç-benzeri veya ince/belirsiz) ve saf piksel-risk sinyali artık
            # `ardisik_onay_esigi` kadar ardışık karede tekrar görülmeden gerçek tehdit sayılmıyor.
            olasi_tehdit = guvenilir_hedef_sayisi > 0 or hareket_adayi_sayisi > 0 or yuzde_risk > 0.10

            if guvenilir_hedef_sayisi > 0:
                st.session_state["ardisik_hareket_sayaci"] = ardisik_onay_esigi  # güvenilir tespit -> anında onay
            elif olasi_tehdit:
                st.session_state["ardisik_hareket_sayaci"] += 1
            else:
                st.session_state["ardisik_hareket_sayaci"] = 0

            tehdit_onaylandi = st.session_state["ardisik_hareket_sayaci"] >= ardisik_onay_esigi

            # Dinamik ihlal ve bildirim tetikleyicisi
            if tehdit_onaylandi:
                rapor_satirlari = []
                if insan_sayisi > 0:
                    rapor_satirlari.append(f"👥 {insan_sayisi} insan görüntülendi.")
                if (arac_sayisi + hareketli_arac_benzeri) > 0:
                    rapor_satirlari.append(f"🚗 {arac_sayisi + hareketli_arac_benzeri} tane araç tespit edildi.")
                if drone_sayisi > 0:
                    rapor_satirlari.append(f"🛸 drone tespit edildi.")
                if hareketli_bilinmeyen_cisim > 0:
                    rapor_satirlari.append(f"🔍 {hareketli_bilinmeyen_cisim} adet sınıflandırılamayan hareket (ince cisim).")

                # 🔧 DÜZELTME: aktif_hedef_sayisi == 0 iken sadece piksel bazlı risk (yuzde_risk) eşiği
                # aşıldığında rapor_satirlari boş kalıyor ve ekranda "TEHDİT:" yazıp arkası boş görünüyordu.
                # Bu genelde nesne henüz sınıflandırılamamış ama hareket/ısı sinyali var demektir
                # (örn. araç henüz ağaç gölgesinde / kısmen görünür durumda).
                if not rapor_satirlari:
                    rapor_satirlari.append(f"🌫️ Sınıflandırılmamış hareket sinyali (%{yuzde_risk:.2f} yoğunluk) - olası gizlenmiş hedef.")

                durum_mesaji = " | ".join(rapor_satirlari)
                durum_ekrani.error(f"🚨 TEHDİT: {durum_mesaji}")

                if "canli_bildirim" not in st.session_state:
                    zaman_damgasi = datetime.now().strftime("%H:%M:%S")
                    yeni_log = f"⏱️ {zaman_damgasi} -> ⚠️ UYARI: {durum_mesaji}"
                    st.session_state["ihlal_gunlugu"].insert(0, yeni_log)

                    detayli_rapor = (
                        f"🚨 <b>AeroGuard Kritik Bölge İhlal Raporu</b> 🚨\n\n"
                        f"⚠️ <b>Durum:</b> Sınır İhlali Gerçekleşti!\n"
                        f"📊 <b>Tehdit Yoğunluğu:</b> %{yuzde_risk:.2f}\n\n"
                        f"<b>Tespit Edilenler:</b>\n"
                        + "\n".join([f"- {satir}" for satir in rapor_satirlari]) + "\n\n"
                        f"🛡️ <b>Güvenlik Seviyesi:</b> KRİTİK REAKSİYON"
                    )
                    telegram_bildirim_gonder(detayli_rapor)
                    st.session_state["canli_bildirim"] = True
            else:
                durum_ekrani.success("🟢 Temiz: Koruma koridorunda şüpheli hareket yok.")
                if "canli_bildirim" in st.session_state:
                    del st.session_state["canli_bildirim"]

            if st.session_state["ihlal_gunlugu"]:
                log_alani.text("\n".join(st.session_state["ihlal_gunlugu"][:10]))

    kamera.release()
else:
    st.info("Canlı tehdit analizi döngüsünü tetiklemek için lütfen 'Kritik Bölge Takibini Başlat' kutucuğunu işaretleyin.")
    
    
    
          