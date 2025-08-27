import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metpy.units import units
from metpy.calc import (
    parcel_profile, cape_cin, lcl,
    lifted_index, k_index, dewpoint_from_relative_humidity,
    most_unstable_cape_cin, mixed_layer_cape_cin,
    wind_components,
    mixing_ratio_from_relative_humidity,
    galvez_davison_index
)
from metpy.plots import SkewT
from datetime import datetime
import requests
import pytz
import warnings

# MetPy uyarılarını gizle
warnings.filterwarnings("ignore", category=RuntimeWarning, module='metpy')

@st.cache_data(show_spinner=False)
def get_weather_data(latitude: float, longitude: float):
    """
    Open-Meteo API'den atmosferik profil verilerini çeker.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        hourly_variables = [
            "temperature_1000hPa", "relative_humidity_1000hPa", "geopotential_height_1000hPa",
            "temperature_975hPa", "relative_humidity_975hPa", "geopotential_height_975hPa",
            "temperature_950hPa", "relative_humidity_950hPa", "geopotential_height_950hPa",
            "temperature_925hPa", "relative_humidity_925hPa", "geopotential_height_925hPa",
            "temperature_900hPa", "relative_humidity_900hPa", "geopotential_height_900hPa",
            "temperature_850hPa", "relative_humidity_850hPa", "geopotential_height_850hPa",
            "temperature_800hPa", "relative_humidity_800hPa", "geopotential_height_800hPa",
            "temperature_700hPa", "relative_humidity_700hPa", "geopotential_height_700hPa",
            "temperature_600hPa", "relative_humidity_600hPa", "geopotential_height_600hPa",
            "temperature_500hPa", "relative_humidity_500hPa", "geopotential_height_500hPa",
            "temperature_400hPa", "relative_humidity_400hPa", "geopotential_height_400hPa",
            "temperature_300hPa", "relative_humidity_300hPa", "geopotential_height_300hPa",
            "temperature_250hPa", "relative_humidity_250hPa", "geopotential_height_250hPa",
            "temperature_200hPa", "relative_humidity_200hPa", "geopotential_height_200hPa",
            "temperature_150hPa", "relative_humidity_150hPa", "geopotential_height_150hPa",
            "temperature_100hPa", "relative_humidity_100hPa", "geopotential_height_100hPa",
            "temperature_70hPa", "relative_humidity_70hPa", "geopotential_height_70hPa",
            "temperature_50hPa", "relative_humidity_50hPa", "geopotential_height_50hPa",
            "temperature_30hPa", "relative_humidity_30hPa", "geopotential_height_30hPa",
            "wind_speed_1000hPa", "wind_direction_1000hPa",
            "wind_speed_975hPa", "wind_direction_975hPa",
            "wind_speed_950hPa", "wind_direction_950hPa",
            "wind_speed_925hPa", "wind_direction_925hPa",
            "wind_speed_900hPa", "wind_direction_900hPa",
            "wind_speed_850hPa", "wind_direction_850hPa",
            "wind_speed_800hPa", "wind_direction_800hPa",
            "wind_speed_700hPa", "wind_direction_700hPa",
            "wind_speed_600hPa", "wind_direction_600hPa",
            "wind_speed_500hPa", "wind_direction_500hPa",
            "wind_speed_400hPa", "wind_direction_400hPa",
            "wind_speed_300hPa", "wind_direction_300hPa",
            "wind_speed_250hPa", "wind_direction_250hPa",
            "wind_speed_200hPa", "wind_direction_200hPa",
            "wind_speed_150hPa", "wind_direction_150hPa",
            "wind_speed_100hPa", "wind_direction_100hPa",
            "wind_speed_70hPa", "wind_direction_70hPa",
            "wind_speed_50hPa", "wind_direction_50hPa",
            "wind_speed_30hPa", "wind_direction_30hPa"
        ]
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(hourly_variables),
            "current": "temperature_2m,relative_humidity_2m,pressure_msl,dew_point_2m",
            "timezone": "auto",
            "forecast_days": 1,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "hourly" not in data:
            st.error("API yanıtı saatlik veri içermiyor.")
            return pd.DataFrame(), {}
        
        hourly_df = pd.DataFrame(data["hourly"])
        hourly_df["time"] = pd.to_datetime(hourly_df["time"]).dt.tz_localize('UTC')
        current_data = data.get("current", {})

        return hourly_df, current_data

    except requests.exceptions.RequestException as e:
        st.error(f"Hata: API'ye bağlanırken bir sorun oluştu. Lütfen konum değerlerini veya internet bağlantınızı kontrol edin. Hata: {e}")
        return pd.DataFrame(), {}
    except ValueError as e:
        st.error(f"Hata: Veri işlenirken bir sorun oluştu. Hata: {e}")
        return pd.DataFrame(), {}

@st.cache_data(show_spinner=False)
def create_profiles(hourly_row):
    """
    API'den gelen verileri kullanarak tüm profilleri oluşturur.
    """
    pressure_levels_hpa = np.array([1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30])
    p_profile_data = pressure_levels_hpa
    t_profile_data = np.array([hourly_row.get(f'temperature_{p}hPa') for p in pressure_levels_hpa])
    rh_profile_data = np.array([hourly_row.get(f'relative_humidity_{p}hPa') for p in pressure_levels_hpa])
    wind_speed_data = np.array([hourly_row.get(f'wind_speed_{p}hPa') for p in pressure_levels_hpa])
    wind_direction_data = np.array([hourly_row.get(f'wind_direction_{p}hPa') for p in pressure_levels_hpa])
    valid_indices = ~np.isnan(t_profile_data) & ~np.isnan(rh_profile_data) & ~np.isnan(p_profile_data)
    p_profile = p_profile_data[valid_indices].astype(np.float64) * units.hPa
    temp_profile = t_profile_data[valid_indices].astype(np.float64) * units.degC
    rh_profile = rh_profile_data[valid_indices].astype(np.float64) * units.percent
    wind_speed = wind_speed_data[valid_indices].astype(np.float64) * units.km / units.hour
    wind_direction = wind_direction_data[valid_indices].astype(np.float64) * units.degrees
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dewpoint_profile = dewpoint_from_relative_humidity(temp_profile, rh_profile)

    return p_profile, temp_profile, dewpoint_profile, wind_speed, wind_direction, rh_profile

def calculate_indices(p_profile, temp_profile, dewpoint_profile, p_start, t_start, td_start, rh_profile):
    """
    Meteorolojik indeksleri hesaplar ve döndürür.
    """
    try:
        lcl_pressure, lcl_temperature = lcl(p_start[0], t_start[0], td_start[0])
        parcel_temp_profile = parcel_profile(p_profile, t_start[0], td_start[0])
        li = lifted_index(p_profile, temp_profile, parcel_temp_profile)
        ki = k_index(p_profile, temp_profile, dewpoint_profile)
        cape_sfc, cin_sfc = cape_cin(p_profile, temp_profile, dewpoint_profile, parcel_profile=parcel_temp_profile)
        mixrat_profile = mixing_ratio_from_relative_humidity(p_profile, temp_profile, rh_profile)
        gdi = galvez_davison_index(p_profile, temp_profile, mixrat_profile, p_start[0])

        return {
            'lcl_pressure': lcl_pressure,
            'lcl_temperature': lcl_temperature,
            'parcel_temp_profile': parcel_temp_profile,
            'li': li,
            'ki': ki,
            'cape_sfc': cape_sfc,
            'cin_sfc': cin_sfc,
            'gdi': gdi
        }
    except Exception as e:
        st.error(f"İndeks hesaplamalarında bir hata oluştu: {e}")
        return None

def plot_skewt(p_profile, temp_profile, dewpoint_profile, parcel_temp_profile, wind_speed, wind_direction, user_lat, user_lon, local_time_for_title, user_pressure_msl):
    """
    Skew-T diyagramını oluşturur.
    """
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)

    skew.plot(p_profile, temp_profile, 'r', linewidth=2, label='Atmosfer Sıcaklığı')
    skew.plot(p_profile, dewpoint_profile, 'g', linewidth=2, label='Atmosfer Çiğ Noktası')
    skew.plot(p_profile, parcel_temp_profile, 'k', linestyle='--', linewidth=2, label='Yüzey Parseli')

    skew.shade_cin(p_profile, temp_profile, parcel_temp_profile)
    skew.shade_cape(p_profile, temp_profile, parcel_temp_profile)

    u_winds, v_winds = wind_components(wind_speed.to('meters/second'), wind_direction)
    mask = ~np.isnan(u_winds.magnitude) & (p_profile.magnitude > 0)
    skew.plot_barbs(p_profile[mask], u_winds[mask], v_winds[mask])

    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()

    skew.ax.set_title(f"Skew-T Diyagramı | Konum: Enlem: {user_lat:.2f}°, Boylam: {user_lon:.2f}°\nZaman: {local_time_for_title.strftime('%H:%M')}", fontsize=14)
    skew.ax.set_xlabel(f"Sıcaklık (°C) / Yüzey Basıncı: {user_pressure_msl:.2f} hPa", fontsize=12)
    skew.ax.set_ylabel('Basınç (hPa)', fontsize=12)
    skew.ax.legend()
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)

    st.pyplot(fig)

# Streamlit Arayüzü
st.title("Atmosferik Profil ve Fırtına Analiz Aracı ⛈️")
st.markdown("""
Bu araç, Open-Meteo API'sinden alınan atmosferik verileri kullanarak **Skew-T Diyagramı** çizerek meteorolojik analizler yapmanızı sağlar.
""")

st.subheader("1. Konum Bilgileri")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        user_lat = st.number_input("Enlem (°)", value=40.90, format="%.2f")
    with col2:
        user_lon = st.number_input("Boylam (°)", value=27.47, format="%.2f")

if st.button("Verileri Çek"):
    with st.spinner("Veriler yükleniyor..."):
        hourly_df, current_data = get_weather_data(user_lat, user_lon)
    if not hourly_df.empty and current_data:
        st.session_state['data_fetched'] = True
        st.session_state['hourly_df'] = hourly_df
        st.session_state['current_data'] = current_data
        st.success("Veriler başarıyla çekildi!")
    else:
        st.error("Veri çekilemedi. Lütfen konum bilgilerini kontrol edin.")

if 'data_fetched' in st.session_state and st.session_state.data_fetched:
    st.subheader("2. Analiz Parametreleri")

    local_timezone = pytz.timezone('Europe/Istanbul')
    current_hour_local = datetime.now(local_timezone).hour
    
    st.info(f"Varsayılan yüzey verileri API'den anlık olarak çekildi. İsterseniz kaydırma çubukları ile bu değerleri değiştirebilirsiniz.")
    
    # Varsayılan değerlerin varlığını kontrol et
    if 'current_data' not in st.session_state:
        # Hata durumunda güvenli varsayılan değerler
        st.session_state.current_data = {
            'temperature_2m': 20.0,
            'relative_humidity_2m': 60.0,
            'pressure_msl': 1013.25
        }
    
    temp_default = st.session_state.current_data.get('temperature_2m', 20.0)
    rh_default = st.session_state.current_data.get('relative_humidity_2m', 60.0)
    pressure_default = st.session_state.current_data.get('pressure_msl', 1013.25)
    
    # Hata oluştuğunda değerin NaN olmaması için kontrol
    temp_default = 20.0 if np.isnan(temp_default) else temp_default
    rh_default = 60.0 if np.isnan(rh_default) else rh_default
    pressure_default = 1013.25 if np.isnan(pressure_default) else pressure_default

    col3, col4 = st.columns(2)
    with col3:
        analysis_hour = st.slider("Analiz Saati (0-23)", min_value=0, max_value=23, value=current_hour_local)
    with col4:
        user_pressure = st.slider(f"Yüzey Basıncı (hPa)", min_value=900.0, max_value=1050.0, value=float(pressure_default), step=0.5, format="%.2f")

    user_temp = st.slider("Yüzey Sıcaklığı (°C)", min_value=-20.0, max_value=50.0, value=float(temp_default), step=0.1, format="%.1f")
    user_rh = st.slider("Yüzey Bağıl Nemi (%)", min_value=0.0, max_value=100.0, value=float(rh_default), step=1.0, format="%.0f")

    if st.button("Analiz Yap ve Diyagramı Çiz"):
        try:
            with st.spinner("Analiz yapılıyor..."):
                analysis_time_local = local_timezone.localize(datetime.now().replace(hour=analysis_hour, minute=0, second=0, microsecond=0))
                analysis_time_utc = analysis_time_local.astimezone(pytz.utc)

                hourly_df = st.session_state.hourly_df
                
                time_diffs = (hourly_df['time'] - analysis_time_utc).abs()
                closest_hour_idx = time_diffs.argmin()
                closest_hourly_data = hourly_df.iloc[closest_hour_idx]

                user_input_data = {
                    'temperature_2m': user_temp,
                    'relative_humidity_2m': user_rh,
                    'pressure_msl': user_pressure
                }

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    user_input_data['dew_point_2m'] = dewpoint_from_relative_humidity(
                        np.array([user_input_data['temperature_2m']]) * units.degC,
                        np.array([user_input_data['relative_humidity_2m']]) * units.percent
                    ).to('degC').magnitude[0]

                local_time_for_title = closest_hourly_data['time'].astimezone(local_timezone)
                
                p_profile, temp_profile, dewpoint_profile, wind_speed, wind_direction, rh_profile = create_profiles(closest_hourly_data)
                
                p_start = np.array([user_input_data['pressure_msl']]).astype(np.float64) * units.hPa
                t_start = np.array([user_input_data['temperature_2m']]).astype(np.float64) * units.degC
                td_start = np.array([user_input_data['dew_point_2m']]).astype(np.float64) * units.degC
                
                indices = calculate_indices(p_profile, temp_profile, dewpoint_profile, p_start, t_start, td_start, rh_profile)
                
                st.subheader("3. Meteorolojik İndeksler")
                if indices:
                    st.write("---")
                    st.markdown(f"**Yükselme İndeksi (LI)**: {indices['li'].magnitude[0]:.2f} °C")
                    if indices['li'].magnitude[0] < 0:
                        st.info("Negatif değerler **kararsızlığı** gösterir. Fırtına olasılığı artar.")
                    else:
                        st.info("Pozitif değerler **kararlılığı** gösterir. Fırtına oluşumu beklenmez.")
                    
                    st.write("---")
                    st.markdown(f"**K-İndeksi (KI)**: {indices['ki'].magnitude:.2f} °C")
                    if indices['ki'].magnitude >= 30:
                        st.warning("Değer 30'un üzerinde: Gök gürültülü fırtına olasılığı yüksektir.")
                    elif 20 <= indices['ki'].magnitude < 30:
                        st.warning("Değer 20-30 arası: Gök gürültülü fırtına olasılığı ortadır.")
                    else:
                        st.info("Değer 20'nin altında: Gök gürültülü fırtına olasılığı düşüktür.")

                    st.write("---")
                    st.markdown(f"**Konvektif Kullanılabilir Potansiyel Enerji (CAPE)**: {indices['cape_sfc'].magnitude:.2f} J/kg")
                    if indices['cape_sfc'].magnitude > 1000:
                        st.warning(f"Yüksek CAPE ({indices['cape_sfc'].magnitude:.0f} J/kg), güçlü fırtına ve sağanak yağış potansiyeline işaret eder.")
                    else:
                        st.info(f"Düşük CAPE ({indices['cape_sfc'].magnitude:.0f} J/kg), atmosferin nispeten kararlı olduğunu gösterir.")
                    
                    st.write("---")
                    st.markdown(f"**Konvektif Engelleme (CIN)**: {indices['cin_sfc'].magnitude:.2f} J/kg")
                    st.info("CIN, atmosferin fırtına gelişimine karşı koyduğu direnç miktarıdır. Değer ne kadar düşükse, fırtına oluşumu o kadar kolaydır.")
                    
                    st.write("---")
                    st.markdown(f"**Galvez-Davison İndeksi (GDI)**: {indices['gdi'].magnitude:.2f}")
                    gdi_val = indices['gdi'].magnitude
                    if gdi_val >= 45:
                        st.warning("Beklenen Konvektif Rejim: Yer yer şiddetli gök gürültülü sağanak yağış bekleniyor.")
                    elif 35 <= gdi_val < 45:
                        st.info("Beklenen Konvektif Rejim: Yer yer gök gürültülü sağanak yağışlar ve/veya yer yer geniş alana yayılmış sağanak yağışlar.")
                    elif 25 <= gdi_val < 35:
                        st.info("Beklenen Konvektif Rejim: Sadece yer yer gök gürültülü sağanak yağışlar ve/veya yer yer sağanak yağışlar.")
                    elif 15 <= gdi_val < 25:
                        st.info("Beklenen Konvektif Rejim: İzole gök gürültülü sağanak yağışlar ve/veya izole sağanak yağışlar.")
                    elif 5 <= gdi_val < 15:
                        st.info("Beklenen Konvektif Rejim: Yer yer sağanak yağışlı.")
                    else:
                        st.info("Beklenen Konvektif Rejim: Kuvvetli TWI muhtemel, hafif yağmur mümkün.")

                    st.subheader("4. Skew-T Diyagramı")
                    plot_skewt(p_profile, temp_profile, dewpoint_profile, indices['parcel_temp_profile'], wind_speed, wind_direction, user_lat, user_lon, local_time_for_title, user_input_data['pressure_msl'])
                
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
