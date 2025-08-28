# Gerekli kütüphaneleri yükleyin
# !pip install -q metpy matplotlib pandas numpy requests pytz

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
import sys

# MetPy uyarılarını gizle
warnings.filterwarnings("ignore", category=RuntimeWarning, module='metpy')

# --- API'den Veri Çekme Fonksiyonu ---
def get_weather_data(latitude: float, longitude: float):
    """
    Open-Meteo API'den atmosferik profil verilerini çeker.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        hourly_variables = [
            "temperature_2m", "relative_humidity_2m", "pressure_msl",
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
            raise ValueError("API yanıtı saatlik veri içermiyor.")
        hourly_df = pd.DataFrame(data["hourly"])
        hourly_df["time"] = pd.to_datetime(hourly_df["time"]).dt.tz_localize('UTC')
        current_data = data.get("current", {})
        return hourly_df, current_data

    except requests.exceptions.RequestException as e:
        print(f"Hata: API'ye bağlanırken bir sorun oluştu. Hata: {e}", file=sys.stderr)
        return pd.DataFrame(), {}
    except ValueError as e:
        print(f"Hata: Veri işlenirken bir sorun oluştu. Hata: {e}", file=sys.stderr)
        return pd.DataFrame(), {}

# --- Profilleri Oluşturma Fonksiyonu ---
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

# --- Ana Program Akışı ---
if __name__ == "__main__":
    try:
        user_lat_input = input("Lütfen enlem bilgisini girin (örn: 40.90): ")
        user_lon_input = input("Lütfen boylam bilgisini girin (örn: 27.47): ")
        
        default_lat = 40.90
        default_lon = 27.47
        user_lat = float(user_lat_input) if user_lat_input.strip() else default_lat
        user_lon = float(user_lon_input) if user_lon_input.strip() else default_lon
    
    except ValueError:
        print("Geçersiz giriş. Lütfen sayısal değerler girin.", file=sys.stderr)
        user_lat, user_lon = 40.90, 27.47
        print(f"Varsayılan değerler kullanılıyor: Enlem: {user_lat}, Boylam: {user_lon}")
    
    print(f"\nAPI'den veri çekiliyor... Konum: Enlem: {user_lat:.2f}°, Boylam: {user_lon:.2f}°")
    hourly_df, current_data = get_weather_data(user_lat, user_lon)

    if not hourly_df.empty and current_data:
        try:
            user_time_input = input("Analiz yapmak istediğiniz saati girin (0-23, boş bırakmak için Enter): ")
            user_temp_input = input("Lütfen yüzey sıcaklığını girin (°C, boş bırakmak için Enter): ")
            user_rh_input = input("Lütfen yüzey bağıl nemini girin (%, boş bırakmak için Enter): ")
            user_pressure_input = input("Lütfen yüzey basıncını girin (hPa, boş bırakmak için Enter): ")
            
            local_timezone = pytz.timezone('Europe/Istanbul')
            if user_time_input.strip():
                try:
                    analysis_hour = int(user_time_input.strip())
                    if not (0 <= analysis_hour <= 23):
                        print("Geçersiz saat girdisi. Anlık saat kullanılacak.")
                        analysis_hour = datetime.now(local_timezone).hour
                except ValueError:
                    print("Geçersiz saat formatı. Anlık saat kullanılacak.", file=sys.stderr)
                    analysis_hour = datetime.now(local_timezone).hour
            else:
                analysis_hour = datetime.now(local_timezone).hour
                print(f"Saat boş bırakıldı. Anlık yerel saat ({analysis_hour:02d}:00) kullanılacak.")
            
            analysis_time_local = local_timezone.localize(datetime.now().replace(hour=analysis_hour, minute=0, second=0, microsecond=0))
            analysis_time_utc = analysis_time_local.astimezone(pytz.utc)
            time_diffs = (hourly_df['time'] - analysis_time_utc).abs()
            closest_hour_idx = time_diffs.argmin()
            closest_hourly_data = hourly_df.iloc[closest_hour_idx]

            user_input_data = {}
            user_input_data['temperature_2m'] = float(user_temp_input) if user_temp_input.strip() else current_data.get('temperature_2m', closest_hourly_data['temperature_2m'])
            user_input_data['relative_humidity_2m'] = float(user_rh_input) if user_rh_input.strip() else current_data.get('relative_humidity_2m', closest_hourly_data['relative_humidity_2m'])
            user_input_data['pressure_msl'] = float(user_pressure_input) if user_pressure_input.strip() else current_data.get('pressure_msl', closest_hourly_data['pressure_msl'])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                user_input_data['dew_point_2m'] = dewpoint_from_relative_humidity(
                    np.array([user_input_data['temperature_2m']]) * units.degC,
                    np.array([user_input_data['relative_humidity_2m']]) * units.percent
                ).to('degC').magnitude[0]

        except (ValueError, KeyError, TypeError) as e:
            print(f"Hata: Kullanıcı verileri işlenirken bir sorun oluştu. {e}. API'nin en yakın verileri kullanılacak.", file=sys.stderr)
            user_input_data = {
                'temperature_2m': current_data.get('temperature_2m', closest_hourly_data['temperature_2m']),
                'relative_humidity_2m': current_data.get('relative_humidity_2m', closest_hourly_data['relative_humidity_2m']),
                'pressure_msl': current_data.get('pressure_msl', closest_hourly_data['pressure_msl']),
                'dew_point_2m': current_data.get('dew_point_2m', dewpoint_from_relative_humidity(
                    np.array([closest_hourly_data['temperature_2m']]) * units.degC,
                    np.array([closest_hourly_data['relative_humidity_2m']]) * units.percent
                ).to('degC').magnitude[0])
            }
        
        local_time_for_title = closest_hourly_data['time'].astimezone(local_timezone)
        print(f"\n--- Analiz ve Diyagram Oluşturuluyor: Seçilen Zaman {local_time_for_title.strftime('%H:%M')} ---")
        p_profile, temp_profile, dewpoint_profile, wind_speed, wind_direction, rh_profile = create_profiles(closest_hourly_data)

        if p_profile.size == 0:
            print("Hata: Atmosferik profil için yeterli veri bulunamadı. Lütfen başka bir saat veya konum deneyin.", file=sys.stderr)
            sys.exit(1)

        p_start = np.array([user_input_data['pressure_msl']]).astype(np.float64) * units.hPa
        t_start = np.array([user_input_data['temperature_2m']]).astype(np.float64) * units.degC
        td_start = np.array([user_input_data['dew_point_2m']]).astype(np.float64) * units.degC
        
        lcl_pressure, lcl_temperature = lcl(p_start[0], t_start[0], td_start[0])
        parcel_temp_profile = parcel_profile(p_profile, t_start[0], td_start[0])

        li = lifted_index(p_profile, temp_profile, parcel_temp_profile)
        ki = k_index(p_profile, temp_profile, dewpoint_profile)
        cape_sfc, cin_sfc = cape_cin(p_profile, temp_profile, dewpoint_profile, parcel_profile=parcel_temp_profile)
        cape_mu, cin_mu = most_unstable_cape_cin(p_profile, temp_profile, dewpoint_profile)
        cape_ml, cin_ml = mixed_layer_cape_cin(p_profile, temp_profile, dewpoint_profile)
        mixrat_profile = mixing_ratio_from_relative_humidity(p_profile, temp_profile, rh_profile)
        gdi = galvez_davison_index(p_profile, temp_profile, mixrat_profile, p_start[0])

        print("\n--- Meteorolojik İndekslerin Anlamları ---")
        print(f"**Kaldırma Yoğunlaşma Seviyesi (LCL)**: Basınç: {lcl_pressure:.2f}, Sıcaklık: {lcl_temperature:.2f}")
        print("   - LCL, bir hava parselinin doygunluğa ulaştığı, yani bulut tabanının oluştuğu seviyedir.")
        print("-" * 50)
        
        print(f"**Yüzey Parseli (SFC) CAPE**: {cape_sfc.magnitude:.2f} J/kg")
        print(f"**Yüzey Parseli (SFC) CIN**: {cin_sfc.magnitude:.2f} J/kg")
        print(f"**En Kararsız Parsel (MU) CAPE**: {cape_mu.magnitude:.2f} J/kg")
        print(f"**En Kararsız Parsel (MU) CIN**: {cin_mu.magnitude:.2f} J/kg")
        print(f"**Karışım Katmanı (ML) CAPE**: {cape_ml.magnitude:.2f} J/kg")
        print(f"**Karışım Katmanı (ML) CIN**: {cin_ml.magnitude:.2f} J/kg")
        print("-" * 50)
        
        # Sadece 500 hPa seviyesindeki LI değerini bulup yazdır
        if 500.0 in p_profile.magnitude.tolist():
            li_500hPa = li[p_profile.magnitude.tolist().index(500.0)]
            print(f"**Yükselme İndeksi (LI)**: {li_500hPa.magnitude:.2f} °C")
            if li_500hPa.magnitude < 0:
                print(f"   - Negatif değerler **kararsızlığı** gösterir. Fırtına olasılığı artar.")
            else:
                print(f"   - Pozitif değerler **kararlılığı** gösterir. Fırtına oluşumu beklenmez.")
        else:
            print("Uyarı: 500 hPa seviyesi profilde bulunamadı. LI değeri hesaplanamadı.")
        print("-" * 50)
        
        print(f"**K-İndeksi (KI)**: {ki.magnitude:.2f} °C")
        print(f"   - K-İndeksi, fırtına ve gök gürültüsü olasılığını belirten bir ölçüttür.")
        if ki.magnitude >= 30:
            print("   - Değer 30'un üzerinde: Gök gürültülü fırtına olasılığı yüksektir.")
        elif 20 <= ki.magnitude < 30:
            print("   - Değer 20-30 arası: Gök gürültülü fırtına olasılığı ortadır.")
        else:
            print("   - Değer 20'nin altında: Gök gürültülü fırtına olasılığı düşüktür.")
        print("-" * 50)
        
        print(f"**Galvez-Davison İndeksi (GDI)**: {gdi.magnitude:.2f}")
        gdi_val = gdi.magnitude
        if gdi_val >= 45:
            print("   - Beklenen Konvektif Rejim: Yer yer şiddetli gök gürültülü sağanak yağış bekleniyor.")
        elif 35 <= gdi_val < 45:
            print("   - Beklenen Konvektif Rejim: Yer yer gök gürültülü sağanak yağışlar ve/veya yer yer geniş alana yayılmış sağanak yağışlar.")
        elif 25 <= gdi_val < 35:
            print("   - Beklenen Konvektif Rejim: Sadece yer yer gök gürültülü sağanak yağışlar ve/veya yer yer sağanak yağışlar.")
        elif 15 <= gdi_val < 25:
            print("   - Beklenen Konvektif Rejim: İzole gök gürültülü sağanak yağışlar ve/veya izole sağanak yağışlar.")
        elif 5 <= gdi_val < 15:
            print("   - Beklenen Konvektif Rejim: Yer yer sağanak yağışlı.")
        else:
            print("   - Beklenen Konvektif Rejim: Kuvvetli TWI muhtemel, hafif yağmur mümkün.")
        print("-" * 50)
        
        fig = plt.figure(figsize=(10, 10))
        skew = SkewT(fig, rotation=45)
        skew.plot(p_profile, temp_profile, 'r', linewidth=2, label='Atmosfer Sıcaklığı')
        skew.plot(p_profile, dewpoint_profile, 'g', linewidth=2, label='Atmosfer Çiğ Noktası')
        skew.plot(p_profile, parcel_temp_profile, 'k', linestyle='--', linewidth=2, label='Yüzey Parseli')
        skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='k', markersize=8, label='LCL')
        skew.shade_cin(p_profile, temp_profile, parcel_temp_profile)
        skew.shade_cape(p_profile, temp_profile, parcel_temp_profile)
        u_winds, v_winds = wind_components(wind_speed.to('meters/second'), wind_direction)
        mask = ~np.isnan(u_winds.magnitude) & (p_profile.magnitude > 0)
        skew.plot_barbs(p_profile[mask], u_winds[mask], v_winds[mask])
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()
        skew.ax.set_title(f"Skew-T Diyagramı | Konum: Enlem: {user_lat:.2f}°, Boylam: {user_lon:.2f}°\nZaman: {local_time_for_title.strftime('%H:%M')}", fontsize=14)
        skew.ax.set_xlabel(f"Sıcaklık (°C) / Yüzey Basıncı: {user_input_data['pressure_msl']:.2f} hPa", fontsize=12)
        skew.ax.set_ylabel('Basınç (hPa)', fontsize=12)
        skew.ax.legend()
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 40)
        plt.show()

    else:
        print("Veri çekilemedi veya eksik veri. Lütfen konum değerlerini veya internet bağlantınızı kontrol edin.", file=sys.stderr)
