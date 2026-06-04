const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function fetchJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "요청 처리에 실패했습니다." }));
    throw new Error(payload.detail || "요청 처리에 실패했습니다.");
  }
  return response.json();
}

export function fetchHealth() {
  return fetchJson("/api/health");
}

export function fetchMonth(year, month, selectedDate, riskMode) {
  const params = new URLSearchParams({
    year: String(year),
    month: String(month),
    riskMode: String(riskMode),
  });
  if (selectedDate) {
    params.set("selectedDate", selectedDate);
  }
  return fetchJson(`/api/calendar/month?${params.toString()}`);
}

export function fetchPrediction(targetDate) {
  return fetchJson(`/api/predictions/${targetDate}`);
}

export function fetchRiskWindow(targetDate) {
  return fetchJson(`/api/risk-window/${targetDate}`);
}
