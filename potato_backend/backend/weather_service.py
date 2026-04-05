"""
天气获取与地名解析服务：调用 Open-Meteo 的 geocoding 与 forecast 接口，
将结果整理为时序模型所需的输入格式（7天日均温、7天日均湿、当天24小时温度）。
"""
from typing import Tuple, List, Optional, Dict
import requests


def geocode_place(name: str, count: int = 1) -> Tuple[Optional[Dict], Optional[str]]:
    """使用 Open-Meteo 的地名解析接口将地名转为经纬度。
    返回 (result_dict, error_message)。result_dict 示例：{"name":..., "latitude":..., "longitude":..., "country":...}
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        resp = requests.get(url, params={"name": name, "count": count, "language": "zh"}, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        results = j.get("results") or []
        if not results:
            return None, f"未找到地名: {name}"
        r = results[0]
        return {
            "name": r.get("name"),
            "latitude": float(r.get("latitude")),
            "longitude": float(r.get("longitude")),
            "country": r.get("country"),
            "timezone": r.get("timezone"),
        }, None
    except Exception as e:
        return None, str(e)


def fetch_weather_for_model(latitude: float, longitude: float) -> Tuple[Optional[List[float]], Optional[str]]:
    """调用 Open-Meteo 的 forecast 接口，获取新模型所需的6维特征。
    返回 (data_list, error_message)。data_list 长度应为6。
    特征顺序：
      [avg_temp_7d, max_temp_7d, min_temp_7d, rh_over_90_hours, total_rainfall_7d, longest_wet_period_hours]
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relativehumidity_2m,precipitation",
        "past_days": 8,
        "timezone": "auto",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        j = resp.json()
        hourly = j.get("hourly") or {}
        times = hourly.get("time") or []
        temps = hourly.get("temperature_2m") or []
        hums = hourly.get("relativehumidity_2m") or []
        rains = hourly.get("precipitation") or []

        if not (times and temps and hums and rains):
            return None, "未从天气 API 获取到足够的小时数据"

        # 按出现顺序收集每天日期（YYYY-MM-DD）
        dates = []
        date_to_indices = {}
        for idx, t in enumerate(times):
            d = t.split("T")[0]
            if d not in date_to_indices:
                date_to_indices[d] = []
                dates.append(d)
            date_to_indices[d].append(idx)

        if len(dates) < 2:
            return None, "天气数据日期不足，无法计算过去 7 天"

        # today 为最后一个日期，前 7 天为其之前的 7 个日期
        today = dates[-1]
        prev_dates = dates[-8:-1] if len(dates) >= 8 else dates[:-1]
        if len(prev_dates) != 7:
            return None, "过去 7 天的数据不足，无法构建输入"

        avg_temps = []
        max_temps = []
        min_temps = []
        rh90_hours = 0
        total_rain = 0.0
        longest_wet = 0

        current_wet = 0
        for d in prev_dates:
            idxs = date_to_indices.get(d, [])
            if not idxs:
                return None, f"日期 {d} 的小时数据缺失"

            tvals = [float(temps[i]) for i in idxs]
            hvals = [float(hums[i]) for i in idxs]
            rval = [float(rains[i]) for i in idxs]

            avg_temps.append(sum(tvals) / len(tvals))
            max_temps.append(max(tvals))
            min_temps.append(min(tvals))

            rh90_hours += sum(1 for v in hvals if v >= 90)
            total_rain += sum(rval)

            for v in hvals:
                if v >= 90:
                    current_wet += 1
                else:
                    longest_wet = max(longest_wet, current_wet)
                    current_wet = 0

        longest_wet = max(longest_wet, current_wet)

        # 模型输入按顺序：6个特征
        data = [
            float(sum(avg_temps) / len(avg_temps)),
            float(max(max_temps)),
            float(min(min_temps)),
            float(rh90_hours),
            float(total_rain),
            float(longest_wet),
        ]

        return data, None
    except Exception as e:
        return None, str(e)
