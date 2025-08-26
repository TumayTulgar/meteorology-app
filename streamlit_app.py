import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metpy.units import units
from metpy.calc import (
    parcel_profile, cape_cin, lcl, lfc, el,
    lifted_index, k_index, dewpoint_from_relative_humidity,
    most_unstable_cape_cin, mixed_layer_cape_cin
)
from metpy.plots import SkewT
import matplotlib.patheffects as pe

# Harita için gerekli
import folium
from streamlit_folium import folium_static

# --- Yorumlama Fonksiyonu ---
def generate_meteorological_comment(analysis_data):
    """
    Verilen meteorolojik indekslere göre detaylı bir yorum metni oluşturur.
    Markdown formatında başlıklar, listeler ve vurgular içerir.
    """
    commentary = []
    commentary.append("## ☁️ Meteorolojik Analiz Özeti ☁️\n")
    commentary.append("---")

    # CAPE YORUMU
    cape = analysis_data['cape']
    if not np.isnan(cape):
        commentary.append(f"### Konvektif Potansiyel Enerji (CAPE): `{cape:.2f} J/kg`")
        if cape > 2500:
            commentary.append("- **Durum:** Çok Yüksek Potansiyel Enerji ⚡")
            commentary.append("- **Anlamı:** Atmosferde olağanüstü miktarda enerji birikimi var. Bu durum, **şiddetli ve organize fırtınaların**, hatta süper hücrelerin gelişimini destekleyebilir. Tornado riski artabilir.")
        elif cape > 1000:
            commentary.append("- **Durum:** Orta-Yüksek Potansiyel Enerji ⛈️")
            commentary.append("- **Anlamı:** Atmosfer kararsızdır ve **orta ila güçlü fırtınaların** oluşumu için yeterli enerji bulunmaktadır. Gök gürültülü sağanak yağışlar ve lokal olarak dolu görülebilir.")
        elif cape > 200:
            commentary.append("- **Durum:** Düşük Potansiyel Enerji 🌦️")
            commentary.append("- **Anlamı:** Kararsızlık sınırlıdır. Oluşacak fırtınaların genellikle **zayıf veya orta kuvvette** olması beklenir. Yerel sağanak yağışlar görülebilir.")
        else:
            commentary.append("- **Durum:** Çok Düşük Potansiyel Enerji  tranquil")
            commentary.append("- **Anlamı:** Atmosfer kararlıdır, ciddi bir konveksiyon (fırtına) oluşumu için yeterli enerji yoktur. Hava genellikle sakindir.")
    else:
        commentary.append("### Konvektif Potansiyel Enerji (CAPE): `Veri Yok`")
        commentary.append("- **Anlamı:** Atmosferik kararsızlık hakkında kesin bir yorum yapılamamaktadır.")
    
    commentary.append("\n---")

    # CIN YORUMU
    cin = analysis_data['cin']
    if not np.isnan(cin):
        commentary.append(f"### Konvektif Engelleme (CIN): `{cin:.2f} J/kg`")
        if cin >= -50:
            commentary.append("- **Durum:** Zayıf veya Yok Denecek Kadar Bastırıcı Katman ✅")
            commentary.append("- **Anlamı:** Parsel yükselişinin önünde belirgin bir engel yoktur. Eğer CAPE mevcutsa, fırtınalar kolayca tetiklenebilir ve gelişebilir.")
        elif cin < -50 and cin >= -200:
            commentary.append("- **Durum:** Orta Kuvvette Bastırıcı Katman ⚠️")
            commentary.append("- **Anlamı:** Fırtına oluşumu için atmosferde bir miktar engelleyici katman var. Bir parselin bu katmanı aşarak yükselebilmesi için güçlü bir tetikleyici mekanizma (örneğin ısınma, cephe geçişi) gereklidir.")
        else: # cin < -200
            commentary.append("- **Durum:** Çok Güçlü Bastırıcı Katman 🚫")
            commentary.append("- **Anlamı:** Atmosferde konveksiyonu (dikey hava hareketini) ciddi şekilde engelleyen çok güçlü bir katman bulunmaktadır. Bu koşullar altında fırtına oluşumu çok zordur.")
    else:
        commentary.append("### Konvektif Engelleme (CIN): `Veri Yok`")
        commentary.append("- **Anlamı:** Bastırıcı katman hakkında kesin bir yorum yapılamamaktadır.")

    commentary.append("\n---")
    
    # LI YORUMU
    li = analysis_data['li']
    if not np.isnan(li):
        commentary.append(f"### Yükselme İndeksi (LI): `{li:.2f} °C`")
        if li < -3:
            commentary.append("- **Durum:** Yüksek Kararsızlık 🔥")
            commentary.append("- **Anlamı:** Atmosfer oldukça kararsızdır. Şiddetli fırtınalar ve güçlü dikey hareketler için uygun koşullar bulunmaktadır.")
        elif li >= -3 and li < 0:
            commentary.append("- **Durum:** Orta Kararsızlık ☁️")
            commentary.append("- **Anlamı:** Atmosfer orta derecede kararsızdır. Orta kuvvette fırtınalar ve sağanak yağışlar beklenebilir.")
        elif li >= 0 and li < 3:
            commentary.append("- **Durum:** Zayıf Kararsızlık veya Kararlı 💧")
            commentary.append("- **Anlamı:** Atmosfer kararlıdır veya çok hafif kararsızdır. Fırtına oluşumu ihtimali düşüktür.")
        else: # li >= 3
            commentary.append("- **Durum:** Kararlı Atmosfer 🌬️")
            commentary.append("- **Anlamı:** Atmosfer kararlıdır. Konvektif fırtına oluşumu için uygun değildir.")
    else:
        commentary.append("### Yükselme İndeksi (LI): `Veri Yok`")
        commentary.append("- **Anlamı:** Yükselme İndeksi verisi bulunamadı.")
        
    commentary.append("\n---")

    # K-İNDEKSI YORUMU
    k_index = analysis_data['k_index']
    if not np.isnan(k_index):
        commentary.append(f"### K-İndeksi: `{k_index:.2f} °C`")
        if k_index > 35:
            commentary.append("- **Durum:** Yüksek Fırtına ve Sağanak İhtimali ⚡☔")
            commentary.append("- **Anlamı:** Hava kütlesinde yüksek nem ve kararsızlık mevcuttur. Gök gürültülü sağanak yağışlar ve şimşek aktivitesi için elverişli koşullar var.")
        elif k_index > 25:
            commentary.append("- **Durum:** Orta Fırtına ve Sağanak İhtimali ⛈️☔")
            commentary.append("- **Anlamı:** Fırtına ve sağanak yağış ihtimali orta düzeydedir. Yerel gök gürültülü sağanaklar beklenebilir.")
        else:
            commentary.append("- **Durum:** Düşük Fırtına ve Sağanak İhtimali 💧☁️")
            commentary.append("- **Anlamı:** Fırtına oluşumu için koşullar zayıftır. Hafif yağışlar veya bulutluluk görülebilir, ancak şiddetli fırtına beklenmez.")
    else:
        commentary.append("### K-İndeksi: `Veri Yok`")
        commentary.append("- **Anlamı:** K-İndeksi verisi bulunamadı.")
        
    return "\n".join(commentary)

def get_value_for_commentary(metpy_obj):
    """MetPy nesnesinden float değeri alır, yoksa np.nan döndürür."""
    if metpy_obj is not None and np.isfinite(metpy_obj.magnitude):
        return float(metpy_obj.magnitude)
    return np.nan

# --- Streamlit Sayfa Yapılandırması ---
st.set_page_config(
    page_title="Meteoroloji Analiz Uygulaması",
    page_icon="⛈️",
    layout="wide", # Geniş düzen
    initial_sidebar_state="expanded"
)

# --- Uygulama Başlığı ve Açıklaması ---
st.title("⛈️ Atmosferik Parsel Simülasyonu ve Skew-T Analizi")
st.markdown("""
Bu uygulama, belirli bir coğrafi konum için anlık atmosferik profil verilerini **Open-Meteo API**'den çekerek meteorolojik analizler sunar. 
Kullanıcılar, yükselen bir parselin başlangıç koşullarını (basınç, sıcaklık, çiğ noktası) manuel olarak ayarlayabilir ve 
atmosferik kararlılık indeksleri (CAPE, CIN, LI, K-İndeksi) ile **Skew-T Log-P diyagramını** görselleştirebilir.
""")
st.markdown("---")

# --- Yan Panel (Sidebar) için Girişler ---
st.sidebar.header("📍 Konum Bilgileri")
user_lat = st.sidebar.number_input("Enlem (°)", value=40.90, format="%.2f", key="sidebar_lat")
user_lon = st.sidebar.number_input("Boylam (°)", value=27.47, format="%.2f", key="sidebar_lon")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2023 Meteoroloji Uygulaması")


# --- API'den Veri Çekme Fonksiyonu ---
@st.cache_data(ttl=3600)
def get_weather_data(latitude: float, longitude: float) -> (pd.DataFrame, dict):
    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry
        
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://api.open-meteo.com/v1/forecast"
        hourly_variables = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
            "surface_pressure",
            "temperature_1000hPa", "relative_humidity_1000hPa", "wind_speed_1000hPa", "wind_direction_1000hPa", "geopotential_height_1000hPa",
            "temperature_975hPa", "relative_humidity_975hPa", "wind_speed_975hPa", "wind_direction_975hPa", "geopotential_height_975hPa",
            "temperature_950hPa", "relative_humidity_950hPa", "wind_speed_950hPa", "wind_direction_950hPa", "geopotential_height_950hPa",
            "temperature_925hPa", "relative_humidity_925hPa", "wind_speed_925hPa", "wind_direction_925hPa", "geopotential_height_925hPa",
            "temperature_900hPa", "relative_humidity_900hPa", "wind_speed_900hPa", "wind_direction_900hPa", "geopotential_height_999hPa",
            "temperature_850hPa", "relative_humidity_850hPa", "wind_speed_850hPa", "wind_direction_850hPa", "geopotential_height_850hPa",
            "temperature_800hPa", "relative_humidity_800hPa", "wind_speed_800hPa", "wind_direction_800hPa", "geopotential_height_800hPa",
            "temperature_700hPa", "relative_humidity_700hPa", "wind_speed_700hPa", "wind_direction_700hPa", "geopotential_height_700hPa",
            "temperature_600hPa", "relative_humidity_600hPa", "wind_speed_600hPa", "wind_direction_600hPa", "geopotential_height_600hPa",
            "temperature_500hPa", "relative_humidity_500hPa", "wind_speed_500hPa", "wind_direction_500hPa", "geopotential_height_500hPa",
            "temperature_400hPa", "relative_humidity_400hPa", "wind_speed_400hPa", "wind_direction_400hPa", "geopotential_height_400hPa",
            "temperature_300hPa", "relative_humidity_300hPa", "wind_speed_300hPa", "wind_direction_300hPa", "geopotential_height_300hPa",
            "temperature_250hPa", "relative_humidity_250hPa", "wind_speed_250hPa", "wind_direction_250hPa", "geopotential_height_250hPa",
            "temperature_200hPa", "relative_humidity_200hPa", "wind_speed_200hPa", "wind_direction_200hPa", "geopotential_height_200hPa",
            "temperature_150hPa", "relative_humidity_150hPa", "wind_speed_150hPa", "wind_direction_150hPa", "geopotential_height_150hPa",
            "temperature_100hPa", "relative_humidity_100hPa", "wind_speed_100hPa", "wind_direction_100hPa", "geopotential_height_100hPa",
            "temperature_70hPa", "relative_humidity_70hPa", "wind_speed_70hPa", "wind_direction_70hPa", "geopotential_height_70hPa",
            "temperature_50hPa", "relative_humidity_50hPa", "wind_speed_50hPa", "wind_direction_50hPa", "geopotential_height_50hPa",
            "temperature_30hPa", "relative_humidity_30hPa", "wind_speed_30hPa", "wind_direction_30hPa", "geopotential_height_30hPa",
        ]
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": hourly_variables,
            "current": ["temperature_2m", "relative_humidity_2m", "pressure_msl", "dew_point_2m", "wind_speed_10m", "wind_direction_10m"],
            "timezone": "auto",
            "forecast_days": 1,
        }
        
        responses = openmeteo.weather_api(url, params=params)
        hourly = responses[0].Hourly()
        hourly_data = {}
        for i, var_name in enumerate(hourly_variables):
            hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()
        hourly_df = pd.DataFrame(data=hourly_data)
        
        current = responses[0].Current()
        current_data = {
            'pressure_msl_current': current.Variables(2).Value(),
            'temperature_2m_current': current.Variables(0).Value(),
            'dew_point_2m_current': current.Variables(3).Value(),
            'relative_humidity_2m_current': current.Variables(1).Value(),
            'wind_speed_10m_current': current.Variables(4).Value(),
            'wind_direction_10m_current': current.Variables(5).Value(),
        }
        
        return hourly_df, current_data

    except Exception as e:
        st.error(f"Hata: Veri çekilirken bir sorun oluştu. Lütfen enlem ve boylamı kontrol edin. Hata: {e}")
        return pd.DataFrame(), {}
        
# API'den veriyi çekin (Bu kısım butonun dışına alındı)
weather_df, current_data = get_weather_data(user_lat, user_lon)

# --- Harita Gösterimi ---
st.subheader(f"Harita üzerinde konum: Tekirdağ, Tekirdağ, Türkiye ({user_lat:.2f}°, {user_lon:.2f}°)")
m = folium.Map(location=[user_lat, user_lon], zoom_start=10)
folium.Marker([user_lat, user_lon], 
              tooltip=f"Lat: {user_lat:.2f}, Lon: {user_lon:.2f}",
              icon=folium.Icon(color='red', icon='cloud')).add_to(m)
folium_static(m, width=700, height=300) # Harita boyutunu ayarlayabilirsiniz.


if not weather_df.empty:
    # --- Anlık Hava Durumu Metrikleri ---
    st.subheader("Anlık Hava Durumu Bilgileri")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Sıcaklık", value=f"{current_data['temperature_2m_current']:.1f} °C", delta="Küresel Model")
    with col2:
        st.metric(label="Çiğ Noktası", value=f"{current_data['dew_point_2m_current']:.1f} °C", delta="Küresel Model")
    with col3:
        st.metric(label="Basınç (Deniz Seviyesi)", value=f"{current_data['pressure_msl_current']:.1f} hPa", delta="Küresel Model")
    with col4:
        st.metric(label="Bağıl Nem", value=f"{current_data['relative_humidity_2m_current']:.1f} %", delta="Küresel Model")

    with st.expander("🛠️ Manuel Başlangıç Değerlerini Düzenle", expanded=True):
        st.info("Yükselen parselin başlangıç değerlerini kaydırıcıları kullanarak seçin.")
        
        # Kullanıcının düzenleyeceği sliderlar
        t_start_manual = st.slider("Parsel Başlangıç Sıcaklığı (°C)", 
                                    min_value=-50.0, 
                                    max_value=50.0, 
                                    value=current_data['temperature_2m_current'], 
                                    step=0.1, key="t_start_manual")
        
        td_start_manual = st.slider("Parsel Başlangıç Çiğ Noktası (°C)", 
                                    min_value=-50.0, 
                                    max_value=t_start_manual, # Çiğ noktası sıcaklıktan büyük olamaz
                                    value=current_data['dew_point_2m_current'], 
                                    step=0.1, key="td_start_manual")
        
        p_start_manual = st.slider("Parsel Başlangıç Basıncı (hPa)", 
                                    min_value=980.0, 
                                    max_value=1050.0, 
                                    value=current_data['pressure_msl_current'], 
                                    step=0.5, key="p_start_manual")
else:
    st.warning("Veri çekilemedi. Manuel girişler için lütfen konum verilerini kontrol edin veya 'Analiz Et' butonuna basmadan önce API'den veri gelmesini bekleyin.")
    # Varsayılan değerler
    t_start_manual, td_start_manual, p_start_manual = 20.0, 10.0, 1013.25

st.markdown("---")
if st.button("🚀 Atmosferi Analiz Et", type="primary"):
    if weather_df.empty:
        st.error("Analiz için hava durumu verisi bulunamadı. Lütfen geçerli bir konum girdiğinizden emin olun.")
    else:
        with st.spinner('Atmosferik veriler analiz ediliyor, lütfen bekleyin...'):
            # 2. Atmosferik profilleri oluşturma
            pressure_levels_hpa = np.array([1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30])
            p_profile_data = np.concatenate([np.array([current_data['pressure_msl_current']]), pressure_levels_hpa])
            p_profile = np.sort(p_profile_data)[::-1].astype(np.float64) * units.hPa
            
            current_hourly_data = weather_df.iloc[0]
            temp_profile_data = np.concatenate([np.array([current_data['temperature_2m_current']]), np.array([current_hourly_data[f'temperature_{p}hPa'] for p in pressure_levels_hpa])])
            temp_profile = temp_profile_data.astype(np.float64) * units.degC
            
            relative_humidity_profile_data = np.concatenate([np.array([current_data['relative_humidity_2m_current']]), np.array([current_hourly_data[f'relative_humidity_{p}hPa'] for p in pressure_levels_hpa])])
            relative_humidity_profile = relative_humidity_profile_data.astype(np.float64) * units.percent
            
            dewpoint_profile = dewpoint_from_relative_humidity(temp_profile, relative_humidity_profile)
            
            # --- Parsel verilerini birimlere dönüştürme (kullanıcının girdiği değerleri kullanıyoruz) ---
            p_start = p_start_manual * units.hPa
            t_start = t_start_manual * units.degC
            td_start = td_start_manual * units.degC
            
            # 4. Parsel simülasyonu ve indeks hesaplamaları
            parcel_temp_profile = parcel_profile(p_profile, t_start, td_start)
            
            # MetPy hesaplamalarında olası None dönüşlerini yakalamak için daha güvenli yaklaşımlar
            try:
                cape, cin = cape_cin(p_profile, temp_profile, dewpoint_profile, parcel_temp_profile)
                if cape.size == 0: cape = np.array([np.nan]) * units.J/units.kg # Boş array dönüşü için
                if cin.size == 0: cin = np.array([np.nan]) * units.J/units.kg # Boş array dönüşü için
            except Exception:
                cape, cin = np.array([np.nan]) * units.J/units.kg, np.array([np.nan]) * units.J/units.kg

            try:
                mu_cape, mu_cin = most_unstable_cape_cin(p_profile, temp_profile, dewpoint_profile)
                if mu_cape.size == 0: mu_cape = np.array([np.nan]) * units.J/units.kg
                if mu_cin.size == 0: mu_cin = np.array([np.nan]) * units.J/units.kg
            except Exception:
                mu_cape, mu_cin = np.array([np.nan]) * units.J/units.kg, np.array([np.nan]) * units.J/units.kg

            try:
                ml_cape, ml_cin = mixed_layer_cape_cin(p_profile, temp_profile, dewpoint_profile)
                if ml_cape.size == 0: ml_cape = np.array([np.nan]) * units.J/units.kg
                if ml_cin.size == 0: ml_cin = np.array([np.nan]) * units.J/units.kg
            except Exception:
                ml_cape, ml_cin = np.array([np.nan]) * units.J/units.kg, np.array([np.nan]) * units.J/units.kg
            
            try:
                li = lifted_index(p_profile, temp_profile, parcel_temp_profile)
                if li.size == 0: li = np.array([np.nan]) * units.degC
            except Exception:
                li = np.array([np.nan]) * units.degC
            
            try:
                k_index_val = k_index(p_profile, temp_profile, dewpoint_profile)
                if k_index_val.size == 0: k_index_val = np.array([np.nan]) * units.degC
            except Exception:
                k_index_val = np.array([np.nan]) * units.degC
            
            try:
                lcl_p, lcl_t = lcl(p_start, t_start, td_start)
            except Exception:
                lcl_p, lcl_t = None, None
                
            try:
                lfc_p, lfc_t = lfc(p_profile, temp_profile, dewpoint_profile, parcel_temp_profile)
            except Exception:
                lfc_p, lfc_t = None, None
                
            try:
                el_p, el_t = el(p_profile, temp_profile, dewpoint_profile, parcel_temp_profile)
            except Exception:
                el_p, el_t = None, None
            
            st.success("Analiz tamamlandı!")

            # --- Sonuçları Streamlit'te Gösterme ---
            st.header("📊 Analiz Sonuçları ve Parametreler")
            
            # Parsel Bilgileri
            st.subheader("💧 Yükselen Parsel Bilgileri")
            st.markdown(f"- **Başlangıç Basıncı:** `{p_start:.2f~P}`")
            st.markdown(f"- **Başlangıç Sıcaklığı:** `{t_start:.2f~P}`")
            st.markdown(f"- **Başlangıç Çiğ Noktası:** `{td_start:.2f~P}`")
            
            st.markdown(f"- **Yükselme Yoğunlaşma Seviyesi (LCL):** `{'{:.2f~P}'.format(lcl_p.to('hPa')) if lcl_p is not None else 'Yok'}`")
            st.markdown(f"- **Serbest Konveksiyon Seviyesi (LFC):** `{'{:.2f~P}'.format(lfc_p.to('hPa')) if lfc_p is not None else 'Yok'}`")
            st.markdown(f"- **Denge Seviyesi (EL):** `{'{:.2f~P}'.format(el_p.to('hPa')) if el_p is not None else 'Yok'}`")

            # İndeksler Tablosu
            st.subheader("📈 Atmosferik Kararlılık İndeksleri")
            index_data = {
                "İndeks": ["CAPE", "CIN", "MU-CAPE", "ML-CAPE", "LI", "K-İndeksi"],
                "Değer": [
                    f"{cape.to('J/kg').magnitude:.2f} J/kg" if not np.isnan(cape.to('J/kg').magnitude) else "Yok",
                    f"{cin.to('J/kg').magnitude:.2f} J/kg" if not np.isnan(cin.to('J/kg').magnitude) else "Yok",
                    f"{mu_cape.to('J/kg').magnitude:.2f} J/kg" if not np.isnan(mu_cape.to('J/kg').magnitude) else "Yok",
                    f"{ml_cape.to('J/kg').magnitude:.2f} J/kg" if not np.isnan(ml_cape.to('J/kg').magnitude) else "Yok",
                    f"{li.magnitude:.2f} °C" if not np.isnan(li.magnitude) else "Yok",
                    f"{k_index_val.magnitude:.2f} °C" if not np.isnan(k_index_val.magnitude) else "Yok",
                ]
            }
            st.table(pd.DataFrame(index_data)) # st.dataframe yerine st.table daha basit görünüyor

            # --- Kendi Meteorolojik Yorum Fonksiyonumuz ---
            st.subheader("🗣️ Detaylı Meteorolojik Yorum")
            st.markdown("---")
            
            analysis_data_for_commentary = {
                'cape': get_value_for_commentary(cape.to('J/kg')),
                'cin': get_value_for_commentary(cin.to('J/kg')),
                'mu_cape': get_value_for_commentary(mu_cape.to('J/kg')),
                'ml_cape': get_value_for_commentary(ml_cape.to('J/kg')),
                'li': get_value_for_commentary(li),
                'k_index': get_value_for_commentary(k_index_val),
            }
            
            meteorological_comment = generate_meteorological_comment(analysis_data_for_commentary)
            st.markdown(meteorological_comment)

            st.markdown("---")
            
            # --- Skew-T Diyagramını Çizme ve Gösterme ---
            st.header("📉 Skew-T Log-P Diyagramı")
            st.markdown("Atmosferik sıcaklık, çiğ noktası ve parsel yolunu gösteren termodinamik diyagram.")
            
            fig = plt.figure(figsize=(12, 12)) # Diyagram boyutu ayarlandı
            skew = SkewT(fig, rotation=45)
            
            # Çevresel atmosfer ve çiğ noktası çizimi
            skew.plot(p_profile, temp_profile, 'red', linewidth=2.5, linestyle='-', label='Atmosfer Sıcaklığı',
                      path_effects=[pe.Stroke(linewidth=3.5, foreground='black'), pe.Normal()])
            skew.plot(p_profile, dewpoint_profile, 'green', linewidth=2.5, linestyle='-', label='Atmosfer Çiğ Noktası',
                      path_effects=[pe.Stroke(linewidth=3.5, foreground='black'), pe.Normal()])
            
            # Parsel yolu çizimi
            skew.plot(p_profile, parcel_temp_profile, 'blue', linestyle='--', linewidth=2, label='Yükselen Parsel (Manuel Başlangıç)',
                      path_effects=[pe.Stroke(linewidth=3, foreground='gray'), pe.Normal()])
            
            # Adyabatik ve karıştırma çizgileri
            skew.plot_dry_adiabats(color='gray', linestyle=':', alpha=0.5)
            skew.plot_moist_adiabats(color='darkgreen', linestyle=':', alpha=0.5)
            skew.plot_mixing_lines(color='brown', linestyle=':', alpha=0.5)
            
            # LCL, LFC, EL noktaları ve etiketleri
            # Etiketler için arka plan gölgeleri eklendi
            if lcl_p is not None and lcl_t is not None:
                skew.plot(lcl_p, lcl_t, 'o', markerfacecolor='black', markeredgecolor='white', markersize=8)
                skew.ax.text(lcl_t.magnitude + 1, lcl_p.magnitude, 'LCL', 
                             fontsize=11, color='white', ha='left', va='center',
                             path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
            
            if lfc_p is not None and lfc_t is not None:
                skew.plot(lfc_p, lfc_t, 'o', markerfacecolor='red', markeredgecolor='white', markersize=8)
                skew.ax.text(lfc_t.magnitude + 1, lfc_p.magnitude, 'LFC', 
                             fontsize=11, color='white', ha='left', va='center',
                             path_effects=[pe.Stroke(linewidth=2, foreground='red'), pe.Normal()])
            
            if el_p is not None and el_t is not None:
                skew.plot(el_p, el_t, 'o', markerfacecolor='blue', markeredgecolor='white', markersize=8)
                skew.ax.text(el_t.magnitude + 1, el_p.magnitude, 'EL', 
                             fontsize=11, color='white', ha='left', va='center',
                             path_effects=[pe.Stroke(linewidth=2, foreground='blue'), pe.Normal()])

            # Rüzgar Barb'ları
            # Rüzgar hızları ve yönleri için ayrı profiller oluşturulmalı
            # Bu kısım Open-Meteo'dan gelen 10m rüzgar ile sınırlı kalabilir veya diğer seviyelerden çekilmelidir.
            # Şu an için sadece 1000hPa ve üstü rüzgar verileri mevcut.
            
            # Wind speed ve direction'ı birimlerle al
            wind_speed_profile_data = np.array([current_hourly_data[f'wind_speed_{p}hPa'] for p in pressure_levels_hpa]) * units.meters / units.second
            wind_direction_profile_data = np.array([current_hourly_data[f'wind_direction_{p}hPa'] for p in pressure_levels_hpa]) * units.degrees
            
            # 1000hPa altındaki ilk seviye için manuel rüzgar verisi ekleyelim
            # Open-Meteo 10m wind_speed_10m ve wind_direction_10m sağlıyor, bunu başlangıç noktasına ekleyebiliriz.
            
            # Skew-T'ye barb çizimi
            # Sadece pressure_levels_hpa için çizelim, current_data'daki 10m rüzgarı diyagramda göstermek için ayrıca ayarlama gerekebilir.
            # Mevcut MetPy SkewT objesinde direkt pressure_profile'ın tamamı ile eşleşen bir rüzgar profili olmalı.
            
            # Basınç profili ve rüzgar profili boyutlarının eşleştiğinden emin olun
            # Eğer 10m rüzgarı da ekleyeceksek, p_profile'a 10m seviyesinin basıncını ekleyip rüzgar_profilleri.append yapmalıyız.
            
            # Şimdilik sadece ana basınç seviyeleri için wind barb ekleyelim.
            wind_p_levels = pressure_levels_hpa * units.hPa
            
            # Veri boyutlarını kontrol edin ve eşleştirin.
            # Rüzgar verilerini MetPy'nin beklediği formata getirin.
            if len(wind_speed_profile_data) == len(wind_direction_profile_data) == len(wind_p_levels):
                 skew.plot_barbs(wind_p_levels, wind_speed_profile_data, wind_direction_profile_data, 
                                 xloc=0.9, # Sağ kenara yakın bir yere yerleştirin
                                 fill_empty_barb=True, 
                                 sizes=dict(emptybarb=0.075, half=0.1, full=0.15, flag=0.15),
                                 barb_kwargs={'color': 'purple', 'linewidth': 1.5})
                 st.markdown("*(Sağ kenardaki mor oklar rüzgar yönü ve hızını göstermektedir.)*")
            else:
                 st.warning("Rüzgar barb'ları çizilemedi: Basınç ve rüzgar verisi boyutları eşleşmiyor.")


            skew.ax.set_title(f'Skew-T Diyagramı (Konum: {user_lat:.2f}, {user_lon:.2f})', fontsize=16, weight='bold')
            skew.ax.set_xlabel('Sıcaklık (°C)', fontsize=12)
            skew.ax.set_ylabel('Basınç (hPa)', fontsize=12)
            skew.ax.legend(loc='upper left')
            skew.ax.set_ylim(1050, 100) # Y ekseni limitleri ayarlandı
            skew.ax.set_xlim(-40, 40)  # X ekseni limitleri ayarlandı
            
            # Izgara görünümünü iyileştirme
            skew.ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)
            st.markdown("---")
            st.info("💡 **İpuçları:** Skew-T diyagramında: Kırmızı çizgi atmosfer sıcaklığını, yeşil çizgi çiğ noktası sıcaklığını, kesik mavi çizgi ise yükselen parselin sıcaklık değişimini gösterir. LCL, LFC ve EL noktaları konvektif seviyeleri işaretler.")
