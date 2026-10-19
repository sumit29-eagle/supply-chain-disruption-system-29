const form = document.getElementById("analysis-form");
const mode = document.getElementById("mode");
const cityWrap = document.getElementById("city-wrap");
const runBtn = document.getElementById("run-btn");
const errorBox = document.getElementById("error-box");

mode.addEventListener("change", () => cityWrap.classList.toggle("hidden", mode.value !== "live"));

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"
  }[ch]));
}
function riskClass(score) {
  return score >= 65 ? "high" : score >= 35 ? "medium" : "low";
}
function statusClass(level) {
  return String(level).toLowerCase();
}
function renderAgents(agents) {
  const symbols = ["◉", "▤", "↗", "⌁"];
  document.getElementById("agent-grid").innerHTML = agents.map((a, i) => `
    <article class="agent-card">
      <div class="agent-head"><div class="agent-icon">${symbols[i] || "◈"}</div>
        <div><div class="agent-name">${escapeHtml(a.name)}</div><div class="agent-role">${escapeHtml(a.role)}</div></div>
      </div>
      <div class="agent-score"><strong>${a.risk}</strong><span>/ 100</span></div>
      <div class="agent-bar"><span class="${riskClass(a.risk)}" style="width:${a.risk}%"></span></div>
      <div class="agent-status">${escapeHtml(a.status)} risk</div>
      <div class="agent-finding">${escapeHtml(a.finding)}</div>
    </article>`).join("");
}
function renderRecommendations(items) {
  document.getElementById("recommendations").innerHTML = items.map(item => `
    <article class="recommendation">
      <span class="priority ${escapeHtml(item.priority.toLowerCase())}">${escapeHtml(item.priority)}</span>
      <div><h4>${escapeHtml(item.title)}</h4><p>${escapeHtml(item.detail)}</p></div>
    </article>`).join("");
}
function renderWeather(w) {
  document.getElementById("weather-source").textContent = w.source.toUpperCase().includes("LIVE") ? "LIVE SIGNAL" : "DEMO";
  document.getElementById("weather-city-result").textContent = w.city || "Unknown";
  document.getElementById("weather-description").textContent = w.weather || "Unavailable";
  document.getElementById("weather-risk").textContent = `${w.weather_risk}/100`;
  document.getElementById("weather-temp").textContent = w.temperature_c == null ? "—" : `${w.temperature_c}°C`;
  document.getElementById("weather-rain").textContent = w.precipitation_mm == null ? "—" : `${w.precipitation_mm} mm`;
  document.getElementById("weather-note").textContent = w.note || "";
}
form.addEventListener("submit", async event => {
  event.preventDefault();
  errorBox.classList.add("hidden");
  runBtn.disabled = true;
  runBtn.innerHTML = "<span>↻</span> Analyzing…";
  const payload = {
    mode: mode.value,
    supplier_delay_days: document.getElementById("supplier_delay_days").value,
    inventory_days: document.getElementById("inventory_days").value,
    transport_delay_days: document.getElementById("transport_delay_days").value,
    demand_spike_pct: document.getElementById("demand_spike_pct").value,
    weather_city: document.getElementById("weather_city").value
  };
  try {
    const response = await fetch("/api/analyze", {
      method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed. Please check the inputs.");
    document.getElementById("overall-score").textContent = data.overall_risk;
    document.getElementById("overall-meter").style.width = `${data.overall_risk}%`;
    const level = document.getElementById("risk-level");
    level.textContent = `${data.risk_level} risk`;
    level.className = `risk-tag ${statusClass(data.risk_level)}`;
    document.getElementById("actions-count").textContent = data.recommendations.length;
    document.getElementById("agents-count").textContent = data.agents.length;
    document.getElementById("updated-time").textContent = new Date(data.generated_at).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
    renderAgents(data.agents);
    renderRecommendations(data.recommendations);
    renderWeather(data.weather);
  } catch (error) {
    errorBox.textContent = error.message || "Unexpected error. Please try again.";
    errorBox.classList.remove("hidden");
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = "<span>↻</span> Run assessment";
  }
});
form.requestSubmit();
