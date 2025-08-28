# Fonksiyon: API'den veri çekme (önbellek olmadan)
def get_weather_data(latitude: float, longitude: float):
    """
    Open-Meteo API'den atmosferik profil verilerini çeker.
    """
    try:
        # Hata Düzeltme: 'api.open-meteors.com' yerine doğru URL
        url = "https://api.open-meteo.com/v1/forecast"
        hourly_variables = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
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
