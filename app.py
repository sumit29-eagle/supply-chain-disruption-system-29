import os
from datetime import datetime, timezone
from statistics import mean

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

DEFAULT_INPUT = {
    "supplier_delay_days": 4,
    "inventory_days": 8,
    "transport_delay_days": 2,
    "demand_spike_pct": 15,
    "weather_city": "Indore",
}

def clamp(value, low=0, high=100):
    return max(low, min(high, int(round(value))))

def demo_signal():
    """Repeatable local demo signals; no external API required."""
    return {
        "source": "Demo scenario",
        "city": "Indore",
        "weather": "Heavy rain (demo)",
        "weather_risk": 72,
        "temperature_c": 29,
        "precipitation_mm": 18.5,
        "note": "Illustrative data only; not a live weather observation.",
    }

def live_weather_signal(city):
    """Use Open-Meteo current conditions as one proxy signal, not shipment tracking."""
    city = (city or "Indore").strip()[:80]
    if not city:
        city = "Indore"
    headers = {"User-Agent": "SupplyChainDisruptionDemo/1.0"}
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=8,
        headers=headers,
    )
    geo.raise_for_status()
    results = geo.json().get("results") or []
    if not results:
        raise ValueError(f"Could not find a location named '{city}'. Try a larger nearby city.")
    place = results[0]
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=8,
        headers=headers,
    )
    weather.raise_for_status()
    current = weather.json().get("current") or {}
    precipitation = float(current.get("precipitation") or 0)
    wind = float(current.get("wind_speed_10m") or 0)
    code = int(current.get("weather_code") or 0)
    # A transparent heuristic proxy, not a calibrated logistics model.
    weather_risk = clamp((precipitation * 2.2) + max(0, wind - 25) * 1.1 + (20 if code >= 60 else 0))
    descriptions = {
        0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Drizzle",
        55: "Dense drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
        71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
        81: "Rain showers", 82: "Heavy rain showers", 95: "Thunderstorm",
        96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
    }
    return {
        "source": "Open-Meteo live weather",
        "city": f"{place.get('name', city)}, {place.get('country', '')}".strip(", "),
        "weather": descriptions.get(code, f"Weather code {code}"),
        "weather_risk": weather_risk,
        "temperature_c": current.get("temperature_2m"),
        "precipitation_mm": precipitation,
        "wind_speed_kmh": wind,
        "observed_at": current.get("time"),
        "note": "Live weather is a proxy risk signal, not live shipment/carrier status.",
    }

def analyze_supply_chain(payload):
    mode = payload.get("mode", "demo")
    if mode not in ("demo", "live"):
        raise ValueError("Mode must be 'demo' or 'live'.")

    try:
        supplier_delay = float(payload.get("supplier_delay_days", DEFAULT_INPUT["supplier_delay_days"]))
        inventory_days = float(payload.get("inventory_days", DEFAULT_INPUT["inventory_days"]))
        transport_delay = float(payload.get("transport_delay_days", DEFAULT_INPUT["transport_delay_days"]))
        demand_spike = float(payload.get("demand_spike_pct", DEFAULT_INPUT["demand_spike_pct"]))
    except (TypeError, ValueError):
        raise ValueError("Numeric fields must contain valid numbers.")

    if supplier_delay < 0 or transport_delay < 0 or inventory_days < 0:
        raise ValueError("Delay and inventory values cannot be negative.")
    if not 0 <= demand_spike <= 300:
        raise ValueError("Demand spike must be between 0 and 300 percent.")
    if supplier_delay > 90 or transport_delay > 90 or inventory_days > 365:
        raise ValueError("Please enter realistic values within the displayed ranges.")

    if mode == "live":
        try:
            weather = live_weather_signal(payload.get("weather_city", "Indore"))
        except requests.RequestException as exc:
            raise RuntimeError("Live weather service could not be reached. Check internet access or switch to Demo mode.") from exc
    else:
        weather = demo_signal()

    # Four transparent, rule-based agents. Scores are heuristics, not trained ML predictions.
    supplier_risk = clamp(supplier_delay * 12 + max(0, demand_spike - 10) * 0.18)
    inventory_risk = clamp(max(0, 14 - inventory_days) * 6 + max(0, demand_spike) * 0.35)
    logistics_risk = clamp(transport_delay * 14 + weather["weather_risk"] * 0.42)
    demand_risk = clamp(demand_spike * 1.15 + max(0, 7 - inventory_days) * 2.5)
    agents = [
        {
            "name": "Supplier Risk Agent",
            "role": "Supplier delay and reliability",
            "risk": supplier_risk,
            "status": "High" if supplier_risk >= 65 else "Medium" if supplier_risk >= 35 else "Low",
            "finding": f"Supplier delay input is {supplier_delay:g} day(s).",
        },
        {
            "name": "Inventory Agent",
            "role": "Stock coverage and shortage exposure",
            "risk": inventory_risk,
            "status": "High" if inventory_risk >= 65 else "Medium" if inventory_risk >= 35 else "Low",
            "finding": f"Current stock coverage is {inventory_days:g} day(s).",
        },
        {
            "name": "Logistics Agent",
            "role": "Transport delays and weather exposure",
            "risk": logistics_risk,
            "status": "High" if logistics_risk >= 65 else "Medium" if logistics_risk >= 35 else "Low",
            "finding": f"Transport delay input is {transport_delay:g} day(s); weather proxy score is {weather['weather_risk']}/100.",
        },
        {
            "name": "Demand Agent",
            "role": "Demand variation and planning pressure",
            "risk": demand_risk,
            "status": "High" if demand_risk >= 65 else "Medium" if demand_risk >= 35 else "Low",
            "finding": f"Demand is {demand_spike:g}% above baseline.",
        },
    ]
    overall = clamp(mean(a["risk"] for a in agents) * 0.75 + max(a["risk"] for a in agents) * 0.25)
    level = "Critical" if overall >= 75 else "High" if overall >= 55 else "Moderate" if overall >= 30 else "Low"

    recommendations = []
    if supplier_risk >= 45:
        recommendations.append({"priority": "High", "title": "Confirm supplier recovery plan", "detail": "Request a committed dispatch date and identify a qualified backup supplier for critical parts."})
    if inventory_risk >= 40 or inventory_days < 7:
        recommendations.append({"priority": "High", "title": "Protect critical stock", "detail": "Prioritize high-value/high-demand SKUs and calculate a short-term replenishment quantity before stock cover falls further."})
    if logistics_risk >= 40:
        recommendations.append({"priority": "Medium", "title": "Review alternate routes", "detail": "Ask the carrier to compare alternate routes and delivery windows; do not reroute without checking cost and capacity."})
    if demand_risk >= 40:
        recommendations.append({"priority": "Medium", "title": "Rebalance demand and allocation", "detail": "Recheck near-term demand, allocate scarce stock to priority orders, and update the replenishment plan."})
    if weather["weather_risk"] >= 60:
        recommendations.append({"priority": "Medium", "title": "Monitor weather exposure", "detail": f"Weather risk proxy is {weather['weather_risk']}/100 for {weather['city']}. Confirm local road/port conditions with the carrier."})
    if not recommendations:
        recommendations.append({"priority": "Low", "title": "Continue routine monitoring", "detail": "No rule crossed the current intervention thresholds. Reassess when supplier, inventory, demand, or transport inputs change."})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "overall_risk": overall,
        "risk_level": level,
        "agents": agents,
        "recommendations": recommendations,
        "weather": weather,
        "inputs": {
            "supplier_delay_days": supplier_delay,
            "inventory_days": inventory_days,
            "transport_delay_days": transport_delay,
            "demand_spike_pct": demand_spike,
        },
        "disclaimer": "Decision-support prototype. Risk scores are transparent rule-based heuristics, not validated predictions. Verify findings before operational decisions.",
    }

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "Supply Chain Disruption System"})

@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        result = analyze_supply_chain(payload)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
