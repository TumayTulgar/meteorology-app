# ÖNEMLİ NOT: Eğer bir 'ImportError' alıyorsanız, Streamlit Cloud ortamınızdaki
# requirements.txt dosyasının metpy kütüphanesinin güncel bir versiyonunu
# içerdiğinden emin olun. Örnek: 'metpy>=1.4.0'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metpy.units import units
from metpy.calc import (
    parcel_profile, cape_cin, lcl,
    lifted_index, k_index, dewpoint_from_relative_humidity,
    wind_components, showalter_index, total_totals_index,
    precipitable_water, freezing_level, bulk_shear
)
from metpy.plots import SkewT
from datetime import datetime
import requests
import pytz
import warnings
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# MetPy ve diğer kütüphane uyarılarını gizle
warnings.filterwarnings("ignore", category=RuntimeWarning, module='metpy')
warnings.filterwarnings("ignore", category=FutureWarning)

# Fonksiyon: API'den veri çekme
def get_weather_data(latitude: float, longitude: float):
    """
    Open-Meteo API'sinden atmosferik profil verilerini çeker.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        
        # Kullanıcı tarafından sağlanan spesifik saatlik değişkenler
        hourly_variables = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
            "temperature_500hPa", "temperature_700hPa", "temperature_850hPa",
            "relative_humidity_500hPa", "relative_humidity_700hPa", "relative_humidity_850hPa",
            "wind_speed_500hPa", "wind_direction_500hPa", "wind_direction_850hPa",
            "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
            "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
            "temperature_80m", "temperature_120m", "temperature_180m",
            "temperature_1000hPa", "temperature_975hPa", "temperature_950hPa", "temperature_925hPa",
            "temperature_900hPa", "temperature_800hPa", "temperature_600hPa", "temperature_400hPa",
            "temperature_300hPa", "temperature_250hPa", "temperature_200hPa", "temperature_150hPa",
            "temperature_100hPa", "temperature_70hPa", "temperature_50hPa", "temperature_30hPa",
            "relative_humidity_1000hPa", "relative_humidity_975hPa", "relative_humidity_950hPa",
            "relative_humidity_925hPa", "relative_humidity_900hPa", "relative_humidity_800hPa",
            "relative_humidity_600hPa", "relative_humidity_400hPa", "relative_humidity_300hPa",
            "relative_humidity_250hPa", "relative_humidity_200hPa", "relative_humidity_150hPa",
            "relative_humidity_100hPa", "relative_humidity_70hPa", "relative_humidity_50hPa",
            "relative_humidity_30hPa",
            "wind_speed_1000hPa", "wind_speed_975hPa", "wind_speed_950hPa", "wind_speed_925hPa",
            "wind_speed_900hPa", "wind_speed_800hPa", "wind_speed_700hPa", "wind_speed_600hPa",
            "wind_speed_400hPa", "wind_speed_300hPa", "wind_speed_250hPa", "wind_speed_200hPa",
            "wind_speed_150hPa", "wind_speed_100hPa", "wind_speed_70hPa", "wind_speed_50hPa",
            "wind_speed_30hPa", "wind_speed_850hPa",
            "wind_direction_1000hPa", "wind_direction_975hPa", "wind_direction_950hPa",
            "wind_direction_925hPa", "wind_direction_900hPa", "wind_direction_800hPa",
            "wind_direction_700hPa", "wind_direction_600hPa", "wind_direction_400hPa",
            "wind_direction_300hPa", "wind_direction_250hPa", "wind_direction_200hPa",
            "wind_direction_150hPa", "wind_direction_100hPa", "wind_direction_70hPa",
            "wind_direction_50hPa", "wind_direction_30hPa",
            "geopotential_height_1000hPa", "geopotential_height_975hPa", "geopotential_height_950hPa",
            "geopotential_height_925hPa", "geopotential_height_900hPa", "geopotential_height_850hPa",
            "geopotential_height_800hPa", "geopotential_height_700hPa", "geopotential_height_600hPa",
            "geopotential_height_500hPa", "geopotential_height_400hPa", "geopotential_height_300hPa",
            "geopotential_height_250hPa", "geopotential_height_200hPa", "geopotential_height_150hPa",
            "geopotential_height_100hPa", "geopotential_height_70hPa", "geopotential_height_50hPa",
            "geopotential_height_30hPa",
            "surface_pressure", "evapotranspiration", "et0_fao_evapotranspiration",
            "vapour_pressure_deficit", "lifted_index", "cape", "convective_inhibition"
        ]

        params = {
            "latitude": latitude, "longitude": longitude,
            "hourly": ",".join(sorted(list(set(hourly_variables)))), 
            "timezone": "auto", 
            "forecast_days": 1,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "hourly" not in data:
            st.error("API yanıtı saatlik veri içermiyor.")
            return pd.DataFrame()
        
        hourly_df = pd.DataFrame(data["hourly"])
        hourly_df["time"] = pd.to_datetime(hourly_df["time"]).dt.tz_localize('UTC')

        return hourly_df

    except requests.exceptions.RequestException as e:
        st.error(f"Hata: API'ye bağlanırken bir sorun oluştu: {e}")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"Hata: Veri işlenirken bir sorun oluştu: {e}")
        return pd.DataFrame()

# Fonksiyon: Profil oluşturma
@st.cache_data(show_spinner="Atmosferik profiller oluşturuluyor...")
def create_profiles(hourly_row):
    """
    API'den gelen verileri kullanarak tüm profilleri oluşturur.
    """
    pressure_levels_hpa = np.array([1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30])
    
    t_profile_data = np.array([hourly_row.get(f'temperature_{p}hPa') for p in pressure_levels_hpa])
    rh_profile_data = np.array([hourly_row.get(f'relative_humidity_{p}hPa') for p in pressure_levels_hpa])
    wind_speed_data = np.array([hourly_row.get(f'wind_speed_{p}hPa') for p in pressure_levels_hpa])
    wind_direction_data = np.array([hourly_row.get(f'wind_direction_{p}hPa') for p in pressure_levels_hpa])
    geopotential_data = np.array([hourly_row.get(f'geopotential_height_{p}hPa') for p in pressure_levels_hpa])

    valid_indices = ~np.isnan(t_profile_data) & ~np.isnan(rh_profile_data) & ~np.isnan(geopotential_data)
    
    p_profile = pressure_levels_hpa[valid_indices].astype(np.float64) * units.hPa
    temp_profile = t_profile_data[valid_indices].astype(np.float64) * units.degC
    rh_profile = rh_profile_data[valid_indices].astype(np.float64) * units.percent
    wind_speed = wind_speed_data[valid_indices].astype(np.float64) * units.km / units.hour
    wind_direction = wind_direction_data[valid_indices].astype(np.float64) * units.degrees
    h_profile = geopotential_data[valid_indices].astype(np.float64) * units.meter
    dewpoint_profile = dewpoint_from_relative_humidity(temp_profile, rh_profile)

    return p_profile, temp_profile, dewpoint_profile, wind_speed, wind_direction, h_profile

# Fonksiyon: İndeks hesaplama
def calculate_indices(p_profile, temp_profile, dewpoint_profile, h_profile, wind_speed, wind_direction, p_start, t_start, td_start):
    """
    Meteorolojik indeksleri hesaplar ve döndürür.
    """
    try:
        u, v = wind_components(wind_speed, wind_direction)
        parcel_temp_profile = parcel_profile(p_profile, t_start[0], td_start[0])
        indices = {}
        
        indices['lcl_pressure'], indices['lcl_temperature'] = lcl(p_start[0], t_start[0], td_start[0])
        indices['li'] = lifted_index(p_profile, temp_profile, parcel_temp_profile)
        indices['ki'] = k_index(p_profile, temp_profile, dewpoint_profile)
        indices['cape_sfc'], indices['cin_sfc'] = cape_cin(p_profile, temp_profile, dewpoint_profile, parcel_temp_profile)
        indices['showalter'] = showalter_index(p_profile, temp_profile, dewpoint_profile)
        indices['total_totals'] = total_totals_index(p_profile, temp_profile, dewpoint_profile)
        indices['pwat'] = precipitable_water(p_profile, dewpoint_profile)
        indices['freezing_level'] = freezing_level(p_profile, temp_profile)

        try:
            shear_u, shear_v = bulk_shear(p_profile, u, v, height=h_profile, depth=6000 * units.meter)
            indices['bulk_shear_0_6km'] = np.sqrt(shear_u**2 + shear_v**2).to('knots')
        except (ValueError, IndexError):
            indices['bulk_shear_0_6km'] = np.nan * units.knots

        indices['parcel_temp_profile'] = parcel_temp_profile
        return indices

    except Exception as e:
        st.error(f"İndeks hesaplamalarında bir hata oluştu: {e}")
        return None

# Fonksiyon: Skew-T diyagramı çizme
def plot_skewt(p_profile, temp_profile, dewpoint_profile, parcel_temp_profile, wind_speed, wind_direction, user_lat, user_lon, local_time_for_title, user_pressure_msl):
    """ Skew-T diyagramını oluşturur. """
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    skew.plot(p_profile, temp_profile, 'r', linewidth=2, label='Sıcaklık Profili')
    skew.plot(p_profile, dewpoint_profile, 'g', linewidth=2, label='Çiğ Noktası Profili')
    if parcel_temp_profile is not None:
        skew.plot(p_profile, parcel_temp_profile, 'k', linestyle='--', linewidth=2, label='Yüzey Parseli Yolu')
        skew.shade_cin(p_profile, temp_profile, parcel_temp_profile)
        skew.shade_cape(p_profile, temp_profile, parcel_temp_profile)
    u_winds, v_winds = wind_components(wind_speed.to('knots'), wind_direction)
    mask = p_profile >= 100 * units.hPa
    skew.plot_barbs(p_profile[mask], u_winds[mask], v_winds[mask])
    skew.plot_dry_adiabats(); skew.plot_moist_adiabats(); skew.plot_mixing_lines()
    skew.ax.set_title(f"Skew-T Diyagramı | {user_lat:.2f}°, {user_lon:.2f}°\nZaman: {local_time_for_title.strftime('%d-%m-%Y %H:%M')}", fontsize=14)
    skew.ax.set_xlabel(f"Sıcaklık (°C) / Yüzey Basıncı: {user_pressure_msl:.2f} hPa", fontsize=12)
    skew.ax.set_ylabel('Basınç (hPa)'); skew.ax.legend(); skew.ax.set_ylim(1050, 100); skew.ax.set_xlim(-40, 40)
    st.pyplot(fig)

# Fonksiyon: API'den veriyi çek ve yüzey parametrelerini güncelle
def reset_and_fetch_api_data():
    """ Seçili konum ve zaman için API'den veriyi çeker ve yüzey parametrelerini günceller. """
    with st.spinner("Güncel veriler çekiliyor..."):
        user_lat, user_lon = st.session_state.coords
        analysis_hour = int(st.session_state.analysis_time_str.split(':')[0])
        hourly_df = get_weather_data(user_lat, user_lon)
        if hourly_df.empty:
            st.warning("Veriler alınamadı. Lütfen tekrar deneyin."); return
        local_timezone = pytz.timezone('Europe/Istanbul')
        analysis_time_local = local_timezone.localize(datetime.now().replace(hour=analysis_hour, minute=0, second=0, microsecond=0))
        analysis_time_utc = analysis_time_local.astimezone(pytz.utc)
        closest_hour_idx = (hourly_df['time'] - analysis_time_utc).abs().idxmin()
        closest_hourly_data = hourly_df.loc[closest_hour_idx]
        st.session_state.user_temp = closest_hourly_data.get('temperature_2m', 20.0)
        st.session_state.user_rh = closest_hourly_data.get('relative_humidity_2m', 60.0)
        st.session_state.user_pressure = closest_hourly_data.get('pressure_msl', 1013.25)

# --- Streamlit Arayüzü ---
st.set_page_config(layout="wide")
st.title("Atmosferik Profil ve Fırtına Analiz Aracı ⛈️")
st.markdown("Bu araç, Open-Meteo API'sinden alınan atmosferik verileri kullanarak **Skew-T Diyagramı** ve çeşitli **meteorolojik indeksler** üreterek fırtına potansiyel analizi yapmanızı sağlar.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Konum ve Zaman Bilgileri")
    if 'coords' not in st.session_state: st.session_state.coords = [40.90, 27.47]
    m = folium.Map(location=st.session_state.coords, zoom_start=10)
    folium.Marker(st.session_state.coords, popup="Seçilen Konum").add_to(m)
    st.markdown("Harita üzerinde analiz yapmak istediğiniz konumu seçin.")
    map_data = st_folium(m, height=300, width=700)
    if map_data and map_data.get("last_clicked"):
        st.session_state.coords = [map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]]
    user_lat, user_lon = st.session_state.coords
    st.write(f"Seçilen Enlem: **{user_lat:.2f}°** | Seçilen Boylam: **{user_lon:.2f}°**")
    local_timezone = pytz.timezone('Europe/Istanbul')
    current_hour_local = datetime.now(local_timezone).hour
    hour_options = [f"{h:02d}:00" for h in range(24)]
    st.session_state.analysis_time_str = st.selectbox("Analiz Saati (Lokal)", options=hour_options, index=current_hour_local)
    analysis_hour = int(st.session_state.analysis_time_str.split(':')[0])

with col2:
    st.subheader("2. Yüzey Parametreleri (İsteğe Bağlı)")
    if 'user_temp' not in st.session_state: st.session_state.user_temp = 20.0
    if 'user_rh' not in st.session_state: st.session_state.user_rh = 60.0
    if 'user_pressure' not in st.session_state: st.session_state.user_pressure = 1013.25
    st.info("Yüzey verilerini manuel olarak değiştirebilir veya 'Sıfırla' butonuyla API verilerini yeniden yükleyebilirsiniz.")
    st.slider("Yüzey Sıcaklığı (°C)", -20.0, 50.0, float(st.session_state.user_temp), 0.1, "%.1f", key="user_temp")
    st.slider("Yüzey Bağıl Nemi (%)", 0.0, 100.0, float(st.session_state.user_rh), 1.0, "%.0f", key="user_rh")
    st.slider("Yüzey Basıncı (hPa)", 900.0, 1050.0, float(st.session_state.user_pressure), 0.5, "%.2f", key="user_pressure")
    st.button("Tüm Yüzey Parametrelerini Sıfırla", key="reset_all", on_click=reset_and_fetch_api_data)

if st.button("⚡ ANALİZ YAP ⚡", use_container_width=True):
    try:
        user_input_data = {'temperature_2m': st.session_state.user_temp, 'relative_humidity_2m': st.session_state.user_rh, 'pressure_msl': st.session_state.user_pressure}
        hourly_df = get_weather_data(user_lat, user_lon)
        if hourly_df.empty: st.error("API'den veri alınamadığı için analiz yapılamıyor."); st.stop()
        analysis_time_local = local_timezone.localize(datetime.now().replace(hour=analysis_hour, minute=0, second=0, microsecond=0))
        analysis_time_utc = analysis_time_local.astimezone(pytz.utc)
        closest_hour_idx = (hourly_df['time'] - analysis_time_utc).abs().idxmin()
        closest_hourly_data = hourly_df.loc[closest_hour_idx]
        user_input_data['dew_point_2m'] = dewpoint_from_relative_humidity(np.array([user_input_data['temperature_2m']])*units.degC, np.array([user_input_data['relative_humidity_2m']])*units.percent).to('degC').magnitude[0]
        local_time_for_title = closest_hourly_data['time'].astimezone(local_timezone)
        p_profile, temp_profile, dewpoint_profile, wind_speed, wind_direction, h_profile = create_profiles(closest_hourly_data)
        p_start, t_start, td_start = np.array([user_input_data['pressure_msl']])*units.hPa, np.array([user_input_data['temperature_2m']])*units.degC, np.array([user_input_data['dew_point_2m']])*units.degC
        indices = calculate_indices(p_profile, temp_profile, dewpoint_profile, h_profile, wind_speed, wind_direction, p_start, t_start, td_start)

        st.subheader("3. Fırtına Potansiyeli Göstergeleri")
        if indices:
            c1, c2, c3 = st.columns(3)
            with c1: st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=indices.get('ki', np.nan*units.degC).magnitude, title={'text': "K-İndeksi (MetPy)"}, gauge={'axis': {'range': [0, 45]}, 'steps': [{'range': [0, 20], 'color': "lightblue"}, {'range': [20, 30], 'color': "yellow"}, {'range': [30, 45], 'color': "red"}]})), use_container_width=True)
            with c2: st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=indices.get('cape_sfc', np.nan*units.J/units.kg).magnitude, title={'text': "CAPE (MetPy)"}, gauge={'axis': {'range': [0, 4000]}, 'steps': [{'range': [0, 1000], 'color': "lightgreen"}, {'range': [1000, 2500], 'color': "yellow"}, {'range': [2500, 4000], 'color': "red"}]})), use_container_width=True)
            with c3: st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=abs(indices.get('cin_sfc', np.nan*units.J/units.kg).magnitude), title={'text': "CIN (MetPy)"}, gauge={'axis': {'range': [0, 200]}, 'steps': [{'range': [0, 25], 'color': "lightgreen"}, {'range': [25, 100], 'color': "yellow"}, {'range': [100, 200], 'color': "red"}]})), use_container_width=True)
        
        st.subheader("4. Ek Yüzey ve Kararsızlık Parametreleri (API)")
        c1, c2, c3 = st.columns(3)
        c1.metric("API CAPE", f"{closest_hourly_data.get('cape', 0):.0f} J/kg")
        c2.metric("API CIN", f"{closest_hourly_data.get('convective_inhibition', 0):.0f} J/kg")
        c3.metric("API Lifted Index", f"{closest_hourly_data.get('lifted_index', 0):.1f} K")
        
        c4, c5, c6 = st.columns(3)
        c4.metric("Yüzey Basıncı", f"{closest_hourly_data.get('surface_pressure', 0):.1f} hPa")
        c5.metric("Buharlaşma-Terleme (ET)", f"{closest_hourly_data.get('evapotranspiration', 0):.3f} mm")
        c6.metric("Buhar Basıncı Açığı", f"{closest_hourly_data.get('vapour_pressure_deficit', 0):.2f} kPa")
        
        st.write("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("5. Detaylı Meteorolojik İndeksler (MetPy)")
            if indices:
                li_val = indices.get('li', [np.nan*units.kelvin])[0].magnitude
                si_val = indices.get('showalter', [np.nan*units.kelvin])[0].magnitude
                tt_val = indices.get('total_totals', np.nan*units.delta_degC).magnitude
                pwat_val = indices.get('pwat', np.nan*units.mm).to('mm').magnitude
                fz_lvl = indices.get('freezing_level', np.nan*units.hPa)
                shear_kts = indices.get('bulk_shear_0_6km', np.nan*units.knots).magnitude
                st.metric("Yükselme İndeksi (LI)", f"{li_val:.2f} °C", "Daha düşük, daha kararsız" if li_val < 0 else "Kararlı")
                st.metric("Showalter İndeksi (SI)", f"{si_val:.2f} °C", "Daha düşük, daha kararsız" if si_val < 0 else "Kararlı")
                st.metric("Total Totals İndeksi (TT)", f"{tt_val:.2f}", "45+ fırtına olasılığı")
                st.metric("Yağış Potansiyeli (PWAT)", f"{pwat_val:.2f} mm", "Yüksek değer, bol nem")
                st.metric("Donma Seviyesi", f"{fz_lvl.to('m').magnitude:.0f} m" if hasattr(fz_lvl, 'magnitude') else "Hesaplanamadı")
                st.metric("0-6 km Rüzgar Kesmesi", f"{shear_kts:.2f} knots", "25+ knots organize fırtına")
        with col2:
            st.subheader("6. Skew-T Diyagramı")
            plot_skewt(p_profile, temp_profile, dewpoint_profile, indices.get('parcel_temp_profile'), wind_speed, wind_direction, user_lat, user_lon, local_time_for_title, user_input_data['pressure_msl'])
    except Exception as e:
        st.error(f"Analiz sırasında beklenmedik bir hata oluştu: {e}")

