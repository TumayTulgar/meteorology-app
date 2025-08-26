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

# --- API İÇİN GEREKLİ KÜTÜPHANELER ---
import requests
import json

def get_value_for_api(metpy_obj):
    """MetPy nesnesinden float değeri alır, yoksa np.nan döndürür."""
    if metpy_obj is not None and np.isfinite(metpy_obj.magnitude):
        return float(metpy_obj.magnitude)
    return np.nan

def ask_gemini_api(data_to_analyze):
    """
    Belirtilen verileri kullanarak Gemini API'ye istek gönderir ve yanıtı döndürür.
    Bu fonksiyon bir şablondur, API anahtarınızı buraya eklemeniz gerekmektedir.
    """
    
    # GERÇEK API ANAHTARINIZI BURAYA EKLEYİN.
    # DİKKAT: BU KODU HERKESE AÇIK BİR PLATFORMA YÜKLERKEN ANAHTARINIZI GİZLEMEYİ UNUTMAYIN.
    API_KEY = "SİZİN_GERÇEK_API_ANAHTARINIZ"  
    
    endpoint = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={API_KEY}"
    
    # f-string içinde her bir değer için ayrı ayrı kontrol yapılıyor
    prompt = f"""
    Aşağıdaki meteorolojik verileri analiz et ve bir cümlelik kısa bir yorum yap. 
    Verilen tüm indeksleri (CAPE, CIN, LI, vb.) ve değerleri dikkate al. 
    Veriler 'nan' ise, o veri için yorum yapma.
    Meteorolojik bilgi düzeyi yüksek, fakat herkesin anlayabileceği şekilde, teknik terimlerden kaçınarak bir özet yaz.
    
    Veriler:
    - Yüzey CAPE: {data_to_analyze['cape']:.2f if not np.isnan(data_to_analyze['cape']) else 'Veri Yok'} J/kg
    - Yüzey CIN: {data_to_analyze['cin']:.2f if not np.isnan(data_to_analyze['cin']) else 'Veri Yok'} J/kg
    - En Kararsız Parsel (MU-CAPE): {data_to_analyze['mu_cape']:.2f if not np.isnan(data_to_analyze['mu_cape']) else 'Veri Yok'} J/kg
    - Karışık Katman Parseli (ML-CAPE): {data_to_analyze['ml_cape']:.2f if not np.isnan(data_to_analyze['ml_cape']) else 'Veri Yok'} J/kg
    - LI (Yükselme İndeksi): {data_to_analyze['li']:.2f if not np.isnan(data_to_analyze['li']) else 'Veri Yok'} Δ°C
    - K-İndeksi: {data_to_analyze['k_index']:.2f if not np.isnan(data_to_analyze['k_index']) else 'Veri Yok'} °C
    """
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
        response.raise_for_status() 
        
        result = response.json()
        
        summary = result['candidates'][0]['content']['parts'][0]['text']
        return summary
    
    except requests.exceptions.RequestException as e:
        return f"API isteği sırasında bir hata oluştu: {e}"
    except (KeyError, IndexError) as e:
        return f"API yanıtı beklenenden farklı. Hata detayı: {e}"
    except Exception as e:
        return f"Genel bir hata oluştu: {e}"

# --- Uygulama Başlığı ve Açıklaması ---
st.title("Atmosferik Parsel Simülasyonu ve Skew-T Analizi")
st.markdown("Bir konumun güncel atmosferik profilini çekerek ve dilerseniz başlangıç verilerini manuel olarak düzenleyerek analiz yapar.")

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
        
# --- Kullanıcıdan Enlem ve Boylam Girişi Alma ---
st.subheader("Konum Bilgileri")
col1, col2 = st.columns(2)
with col1:
    user_lat = st.number_input("Enlem (°)", value=40.90, format="%.2f")
with col2:
    user_lon = st.number_input("Boylam (°)", value=27.47, format="%.2f")

# API'den veriyi çekin (Bu kısım butonun dışına alındı)
weather_df, current_data = get_weather_data(user_lat, user_lon)

if not weather_df.empty:
    with st.expander("Manuel Başlangıç Değerlerini Düzenle"):
        st.info("Yükselen parselin başlangıç değerlerini kaydırıcıları kullanarak seçin.")
        
        # Küresel modelden gelen değerleri göster
        st.write(f"**Küresel modelden gelen değerler (2m):**")
        st.write(f"  - Basınç: **{current_data['pressure_msl_current']:.2f} hPa**")
        st.write(f"  - Sıcaklık: **{current_data['temperature_2m_current']:.2f} °C**")
        st.write(f"  - Çiğ Noktası: **{current_data['dew_point_2m_current']:.2f} °C**")
        
        st.markdown("---")
        
        # Kullanıcının düzenleyeceği sliderlar
        t_start_manual = st.slider("Parsel Başlangıç Sıcaklığı (°C)", 
                                    min_value=-50.0, 
                                    max_value=50.0, 
                                    value=current_data['temperature_2m_current'], 
                                    step=0.1)
        
        # Çiğ noktası sıcaklıktan büyük olamaz
        td_start_manual = st.slider("Parsel Başlangıç Çiğ Noktası (°C)", 
                                    min_value=-50.0, 
                                    max_value=t_start_manual, 
                                    value=current_data['dew_point_2m_current'], 
                                    step=0.1)
        
        p_start_manual = st.slider("Parsel Başlangıç Basıncı (hPa)", 
                                    min_value=980.0, 
                                    max_value=1050.0, 
                                    value=current_data['pressure_msl_current'], 
                                    step=0.5)
else:
    st.warning("Veri çekilemedi. Manuel girişler için lütfen konum verilerini kontrol edin.")
    # Varsayılan değerler
    t_start_manual, td_start_manual, p_start_manual = 20.0, 10.0, 1013.25

if st.button("Analiz Et"):
    if weather_df.empty:
        st.warning("Lütfen önce geçerli konum verilerini girerek verileri güncelleyin.")
    else:
        st.markdown("---")
        st.info(f"Analiz için konum: **Enlem: {user_lat:.2f}°**, **Boylam: {user_lon:.2f}°**")
        
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
        
        cape, cin = cape_cin(p_profile, temp_profile, dewpoint_profile, parcel_temp_profile)
        mu_cape, mu_cin = most_unstable_cape_cin(p_profile, temp_profile, dewpoint_profile)
        ml_cape, ml_cin = mixed_layer_cape_cin(p_profile, temp_profile, dewpoint_profile)
        li = lifted_index(p_profile, temp_profile, parcel_temp_profile)
        k_index_val = k_index(p_profile, temp_profile, dewpoint_profile)
        
        # Hata kontrolü için try-except blokları ekleyelim.
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
            
        # --- Sonuçları Streamlit'te Gösterme ---
        st.header("Analiz Sonuçları")
        st.subheader("Parsel Simülasyonu ve Termodinamik Parametreler")
        st.write(f"**Yükselen Parselin Başlangıç Basıncı:** {p_start:.2f~P}")
        st.write(f"**Yükselen Parselin Başlangıç Sıcaklığı:** {t_start:.2f~P}")
        st.write(f"**Yükselen Parselin Başlangıç Çiğ Noktası:** {td_start:.2f~P}")
        st.write(f"**Yükselme Yoğunlaşma Seviyesi (LCL) Basıncı:** {'{:.2f~P}'.format(lcl_p.to('hPa')) if lcl_p is not None else 'Yok'}")
        
        st.subheader("Konvektif Seviyeler ve İndeksler")
        st.write(f"**Serbest Konveksiyon Seviyesi (LFC) Basıncı:** {'{:.2f~P}'.format(lfc_p.to('hPa')) if lfc_p is not None else 'Yok'}")
        st.write(f"**Denge Seviyesi (EL) Basıncı:** {'{:.2f~P}'.format(el_p.to('hPa')) if el_p is not None else 'Yok'}")
        st.write(f"**Yüzeyden Yükselen Parsel için CAPE:** {cape:.2f~P}")
        st.write(f"**Yüzeyden Yükselen Parsel için CIN:** {cin:.2f~P}")
        st.write(f"**En Kararsız Parsel (MU-CAPE):** {mu_cape:.2f~P}")
        st.write(f"**Karışık Katman Parseli (ML-CAPE):** {ml_cape:.2f~P}")
        st.write(f"**Yükselme İndeksi (LI):** {li:.2f~P}")
        st.write(f"**K-İndeksi:** {k_index_val:.2f~P}")
        
        # --- API İLE YORUMLAMA BÖLÜMÜ ---
        st.subheader("Meteorolojik Durum Özeti (AI Destekli)")
        st.markdown("---")
        
        # Hataları önlemek için değerleri float'a dönüştürüp nan kontrolü yapıyoruz.
        analysis_data = {
            'cape': get_value_for_api(cape.to('J/kg')),
            'cin': get_value_for_api(cin.to('J/kg')),
            'mu_cape': get_value_for_api(mu_cape.to('J/kg')),
            'ml_cape': get_value_for_api(ml_cape.to('J/kg')),
            'li': get_value_for_api(li),
            'k_index': get_value_for_api(k_index_val),
        }
        
        with st.spinner('Yapay zeka analiz yapıyor, lütfen bekleyin...'):
            ai_yorum = ask_gemini_api(analysis_data)
        
        st.write(ai_yorum)

        st.markdown("---")
        
        # --- Skew-T Diyagramını Çizme ve Gösterme ---
        st.header("Skew-T Diyagramı")
        
        fig = plt.figure(figsize=(14, 14))
        skew = SkewT(fig, rotation=45)
        
        skew.plot(p_profile, temp_profile, 'r', linewidth=2, label='Atmosfer Sıcaklığı')
        skew.plot(p_profile, dewpoint_profile, 'g', linewidth=2, label='Atmosfer Çiğ Noktası')
        skew.plot(p_profile, parcel_temp_profile, 'k', linestyle='--', linewidth=2, label='Yükselen Parsel (Manuel Başlangıç)')
        
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()
        
        # Çizim ve etiketler için daha güvenilir kontrol
        if lcl_p is not None and lcl_t is not None:
            skew.plot(lcl_p, lcl_t, 'ko', markerfacecolor='black', markersize=8)
            skew.ax.text(lcl_t.magnitude + 2, lcl_p.magnitude, 'LCL', fontsize=11, color='black', ha='left', va='center')
        
        if lfc_p is not None and lfc_t is not None:
            skew.plot(lfc_p, lfc_t, 'ro', markerfacecolor='red', markersize=8)
            skew.ax.text(lfc_t.magnitude + 2, lfc_p.magnitude, 'LFC', fontsize=11, color='red', ha='left', va='center')
        
        if el_p is not None and el_t is not None:
            skew.plot(el_p, el_t, 'bo', markerfacecolor='blue', markersize=8)
            skew.ax.text(el_t.magnitude + 2, el_p.magnitude, 'EL', fontsize=11, color='blue', ha='left', va='center')
        
        skew.ax.set_title(f'Skew-T Diyagramı (Konum: {user_lat:.2f}, {user_lon:.2f})', fontsize=16)
        skew.ax.set_xlabel('Sıcaklık (°C)', fontsize=12)
        skew.ax.set_ylabel('Basınç (hPa)', fontsize=12)
        skew.ax.legend()
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 40)
        
        st.pyplot(fig)
