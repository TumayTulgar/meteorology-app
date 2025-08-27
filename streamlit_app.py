import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metpy.units import units
from metpy.calc import (
    parcel_profile, cape_cin, lcl,
    lifted_index, k_index, dewpoint_from_relative_humidity,
    wind_components,
)
from metpy.plots import SkewT
from datetime import datetime
import requests
import pytz
import warnings
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go 

# MetPy uyarılarını gizle
warnings.filterwarnings("ignore", category=RuntimeWarning, module='metpy')

# Fonksiyon: API'den veri çekme (önbellek olmadan)
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
            "wind_direction_850hPa", "pressure_msl",
            "wind_direction_30hPa"
        ]
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(hourly_variables),
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
        st.error(f"Hata: API'ye bağlanırken bir sorun oluştu. Lütfen konum değerlerini veya internet bağlantınızı kontrol edin. Hata: {e}")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"Hata: Veri işlenirken bir sorun oluştu. Hata: {e}")
        return pd.DataFrame()

# Fonksiyon: Profil oluşturma (önbellek ile)
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

# Fonksiyon: İndeks hesaplama
def calculate_indices(p_profile, temp_profile, dewpoint_profile, p_start, t_start, td_start):
    """
    Meteorolojik indeksleri hesaplar ve döndürür.
    """
    try:
        lcl_pressure, lcl_temperature = lcl(p_start[0], t_start[0], td_start[0])
        parcel_temp_profile = parcel_profile(p_profile, t_start[0], td_start[0])
        li = lifted_index(p_profile, temp_profile, parcel_temp_profile)
        ki = k_index(p_profile, temp_profile, dewpoint_profile)
        cape_sfc, cin_sfc = cape_cin(p_profile, temp_profile, dewpoint_profile, parcel_profile=parcel_temp_profile)

        return {
            'lcl_pressure': lcl_pressure,
            'lcl_temperature': lcl_temperature,
            'parcel_temp_profile': parcel_temp_profile,
            'li': li,
            'ki': ki,
            'cape_sfc': cape_sfc,
            'cin_sfc': cin_sfc
        }
    except Exception as e:
        st.error(f"İndeks hesaplamalarında bir hata oluştu: {e}")
        return None

# Fonksiyon: Skew-T diyagramı çizme
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

# Fonksiyon: API'den veriyi çek ve yüzey parametrelerini güncelle
def reset_and_fetch_api_data():
    """
    Seçili konum ve zaman için API'den veriyi çeker ve yüzey parametrelerini günceller.
    """
    with st.spinner("Güncel veriler çekiliyor..."):
        user_lat = st.session_state.coords[0]
        user_lon = st.session_state.coords[1]
        analysis_hour = int(st.session_state.analysis_time_str.split(':')[0])

        # API'den tüm saatlik verileri çek
        hourly_df = get_weather_data(user_lat, user_lon)
        
        if hourly_df.empty:
            st.warning("Veriler alınamadı. Lütfen tekrar deneyin.")
            return

        # En yakın saate ait veriyi bul
        local_timezone = pytz.timezone('Europe/Istanbul')
        analysis_time_local = local_timezone.localize(datetime.now().replace(hour=analysis_hour, minute=0, second=0, microsecond=0))
        analysis_time_utc = analysis_time_local.astimezone(pytz.utc)
        time_diffs = (hourly_df['time'] - analysis_time_utc).abs()
        closest_hour_idx = time_diffs.argmin()
        closest_hourly_data = hourly_df.iloc[closest_hour_idx]

        # Yüzey parametrelerini seçili saate ait verilerle güncelle
        st.session_state.user_temp = closest_hourly_data.get('temperature_2m', 20.0)
        st.session_state.user_rh = closest_hourly_data.get('relative_humidity_2m', 60.0)
        st.session_state.user_pressure = closest_hourly_data.get('pressure_msl', 1013.25)


# Streamlit Arayüzü
st.title("Atmosferik Profil ve Fırtına Analiz Aracı ⛈️")
st.markdown("""
Bu araç, Open-Meteo API'sinden alınan atmosferik verileri kullanarak **Skew-T Diyagramı** çizerek meteorolojik analizler yapmanızı sağlar.
""")

st.subheader("1. Konum ve Zaman Bilgileri")

initial_coords = [40.90, 27.47]
if 'coords' not in st.session_state:
    st.session_state.coords = initial_coords

m = folium.Map(location=st.session_state.coords, zoom_start=12)
folium.Marker(st.session_state.coords, popup="Seçilen Konum").add_to(m)

st.markdown("Harita üzerinde analiz yapmak istediğiniz konumu seçin.")
map_data = st_folium(m, height=350, width=700)

if map_data and map_data.get("last_clicked"):
    user_lat = map_data["last_clicked"]["lat"]
    user_lon = map_data["last_clicked"]["lng"]
    st.session_state.coords = [user_lat, user_lon]
else:
    user_lat = st.session_state.coords[0]
    user_lon = st.session_state.coords[1]

st.write(f"Seçilen Enlem: **{user_lat:.2f}°**")
st.write(f"Seçilen Boylam: **{user_lon:.2f}°**")

local_timezone = pytz.timezone('Europe/Istanbul')
current_hour_local = datetime.now(local_timezone).hour
hour_options = [f"{h:02d}:00" for h in range(14, 24)]
default_hour_str = f"{current_hour_local:02d}:00" if 14 <= current_hour_local <= 23 else "14:00"
st.session_state.analysis_time_str = st.selectbox("Analiz Saati", options=hour_options, index=hour_options.index(default_hour_str))
analysis_hour = int(st.session_state.analysis_time_str.split(':')[0])

st.subheader("2. Yüzey Parametreleri (İsteğe Bağlı)")

# session_state'te varsayılan değerleri kontrol et ve ata
if 'user_temp' not in st.session_state:
    st.session_state.user_temp = 20.0
if 'user_rh' not in st.session_state:
    st.session_state.user_rh = 60.0
if 'user_pressure' not in st.session_state:
    st.session_state.user_pressure = 1013.25

st.info("Kaydırma çubukları ile yüzey verilerini değiştirebilirsiniz. Herhangi bir 'Sıfırla' butonuna basarak, seçili konum ve saate ait güncel API verilerini çekebilirsiniz.")

# Her bir parametre için slider ve butonu yan yana koy
temp_col, temp_btn_col = st.columns([0.7, 0.3])
with temp_col:
    st.slider(
        "Yüzey Sıcaklığı (°C)", 
        min_value=-20.0, 
        max_value=50.0, 
        value=float(st.session_state.user_temp), 
        step=0.1, 
        format="%.1f",
        key="user_temp"
    )
with temp_btn_col:
    st.markdown("<br>", unsafe_allow_html=True)  # Hizalama için
    st.button("Sıfırla", key="reset_temp", on_click=reset_and_fetch_api_data)

rh_col, rh_btn_col = st.columns([0.7, 0.3])
with rh_col:
    st.slider(
        "Yüzey Bağıl Nemi (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=float(st.session_state.user_rh), 
        step=1.0, 
        format="%.0f",
        key="user_rh"
    )
with rh_btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Sıfırla", key="reset_rh", on_click=reset_and_fetch_api_data)

pressure_col, pressure_btn_col = st.columns([0.7, 0.3])
with pressure_col:
    st.slider(
        "Yüzey Basıncı (hPa)", 
        min_value=900.0, 
        max_value=1050.0, 
        value=float(st.session_state.user_pressure), 
        step=0.5, 
        format="%.2f",
        key="user_pressure"
    )
with pressure_btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Sıfırla", key="reset_pressure", on_click=reset_and_fetch_api_data)


if st.button("Analiz Yap"):
    try:
        # Analiz için gerekli verileri session_state'ten al
        user_input_data = {
            'temperature_2m': st.session_state.user_temp,
            'relative_humidity_2m': st.session_state.user_rh,
            'pressure_msl': st.session_state.user_pressure
        }
        
        # Sadece analiz için API'den veri çek
        with st.spinner("Analiz için atmosferik profiller oluşturuluyor..."):
            hourly_df = get_weather_data(user_lat, user_lon)
            if hourly_df.empty:
                st.error("API'den veri alınamadığı için analiz yapılamıyor.")
                st.stop()

            local_timezone = pytz.timezone('Europe/Istanbul')
            analysis_time_local = local_timezone.localize(datetime.now().replace(hour=analysis_hour, minute=0, second=0, microsecond=0))
            analysis_time_utc = analysis_time_local.astimezone(pytz.utc)

            time_diffs = (hourly_df['time'] - analysis_time_utc).abs()
            closest_hour_idx = time_diffs.argmin()
            closest_hourly_data = hourly_df.iloc[closest_hour_idx]

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
            
            indices = calculate_indices(p_profile, temp_profile, dewpoint_profile, p_start, t_start, td_start)

            st.subheader("3. Fırtına Potansiyeli Göstergeleri")

            if indices:
                # Kolonlar oluşturma
                col1, col2, col3 = st.columns(3)

                # KI Göstergesi
                with col1:
                    ki_value = indices['ki'].magnitude
                    fig_ki = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=ki_value,
                        title={'text': "K-İndeksi"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, 50], 'tickwidth': 1},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 15], 'color': "green"},
                                {'range': [15, 25], 'color': "yellow"},
                                {'range': [25, 35], 'color': "orange"},
                                {'range': [35, 50], 'color': "red"}]}))
                    st.plotly_chart(fig_ki, use_container_width=True)

                # CAPE Göstergesi
                with col2:
                    cape_value = indices['cape_sfc'].magnitude
                    fig_cape = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=cape_value,
                        title={'text': "CAPE"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, 4000], 'tickwidth': 1},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 500], 'color': "green"},
                                {'range': [500, 1500], 'color': "yellow"},
                                {'range': [1500, 3000], 'color': "orange"},
                                {'range': [3000, 4000], 'color': "red"}]}))
                    st.plotly_chart(fig_cape, use_container_width=True)
                
                # CIN Göstergesi
                with col3:
                    cin_value = indices['cin_sfc'].magnitude
                    fig_cin = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=abs(cin_value),
                        title={'text': "CIN"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 300], 'tickwidth': 1},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "red"},
                                {'range': [50, 200], 'color': "yellow"},
                                {'range': [200, 300], 'color': "green"}]}))
                    st.plotly_chart(fig_cin, use_container_width=True)


            st.write("---")

            st.subheader("4. Detaylı Meteorolojik İndeksler")
            if indices:
                # Yükselme İndeksi (LI)
                li_value = indices['li'].magnitude[0]
                st.markdown(f"**Yükselme İndeksi (LI)**: {li_value:.2f} °C")
                if li_value < 0:
                    st.info("Negatif değerler **kararsızlığı** gösterir. Fırtına olasılığı artar.")
                else:
                    st.success("Pozitif değerler **kararlılığı** gösterir. Fırtına oluşumu beklenmez.")

                st.write("---")

                # K-İndeksi (KI)
                st.markdown(f"**K-İndeksi (KI)**: {ki_value:.2f} °C")
                if ki_value >= 35:
                    st.error("Çok Yüksek Fırtına Potansiyeli. Çok kuvvetli yağış ve fırtına ihtimali yüksek.")
                elif ki_value >= 25:
                    st.warning("Yüksek Fırtına Potansiyeli. Gök gürültülü fırtına ve sağanak yağış ihtimali var.")
                elif ki_value >= 15:
                    st.info("Orta Fırtına Potansiyeli. Hafif gök gürültülü fırtına görülebilir.")
                else:
                    st.success("Düşük Fırtına Potansiyeli.")

                st.write("---")

                # Konvektif Kullanılabilir Potansiyel Enerji (CAPE)
                cape_value = indices['cape_sfc'].magnitude
                st.markdown(f"**Konvektif Kullanılabilir Potansiyel Enerji (CAPE)**: {cape_value:.2f} J/kg")
                if cape_value > 3000:
                    st.error(f"Çok Yüksek CAPE. Çok şiddetli fırtına, dolu ve fırtına rüzgarları gibi tehlikeler görülebilir.")
                elif cape_value > 1500:
                    st.warning(f"Yüksek CAPE. Şiddetli gök gürültülü fırtına ihtimali mevcut.")
                elif cape_value > 500:
                    st.info(f"Orta CAPE. Gök gürültülü fırtına olasılığı mevcut.")
                else:
                    st.success(f"Düşük CAPE. Fırtına potansiyeli düşüktür.")

                st.write("---")

                # Konvektif Engelleme (CIN)
                cin_value = indices['cin_sfc'].magnitude
                st.markdown(f"**Konvektif Engelleme (CIN)**: {cin_value:.2f} J/kg")
                if cin_value > 200:
                    st.success("Yüksek Engelleme. Fırtına oluşumu zorlaşır.")
                elif cin_value > 50:
                    st.info("Orta Engelleme. Fırtına oluşumu için daha güçlü bir tetikleyici gerekebilir.")
                else:
                    st.error("Düşük Engelleme. Atmosfer kolayca kararsız hale gelebilir ve fırtına oluşumu kolaylaşır.")
                
                # --- Skew-T Diyagramı ---
                st.subheader("5. Skew-T Diyagramı")
                plot_skewt(p_profile, temp_profile, dewpoint_profile, indices['parcel_temp_profile'], wind_speed, wind_direction, user_lat, user_lon, local_time_for_title, user_input_data['pressure_msl'])
            
    except Exception as e:
        st.error(f"Analiz sırasında bir hata oluştu: {e}")
