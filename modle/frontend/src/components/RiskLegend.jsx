const ITEMS = [
  { label: "한산함", color: "#fff7cc" },
  { label: "보통", color: "#ffd166" },
  { label: "붐빔", color: "#ff7a3d" },
  { label: "매우 붐빔", color: "#e60000" },
];

export default function RiskLegend({ active, onToggle }) {
  return (
    <div className="risk-legend-card">
      <div className="risk-legend-header">
        <div>
          <p className="panel-kicker">달력 위험도</p>
          <h3>주변 5일 보기</h3>
        </div>
        <button type="button" className={`risk-toggle ${active ? "active" : ""}`} onClick={onToggle}>
          {active ? "끄기" : "위험도 보기"}
        </button>
      </div>
      <div className="risk-legend-row">
        {ITEMS.map((item) => (
          <div key={item.label} className="legend-item">
            <span className="legend-swatch" style={{ backgroundColor: item.color }} />
            <span>{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
