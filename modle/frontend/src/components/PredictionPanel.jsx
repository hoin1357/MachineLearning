function hexToRgba(hex, alpha) {
  const normalized = hex.replace("#", "");
  const red = parseInt(normalized.slice(0, 2), 16);
  const green = parseInt(normalized.slice(2, 4), 16);
  const blue = parseInt(normalized.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function getHolidayLabel(prediction) {
  if (prediction.isHoliday) {
    return "공휴일";
  }
  if (prediction.isWeekend) {
    return "주말";
  }
  return "평일";
}

function getSimpleWeatherLabel(precipitation) {
  if (precipitation >= 10) {
    return "폭우 🌧️";
  }
  if (precipitation > 0) {
    return "약간의 비 🌦️";
  }
  return "쨍쨍함 ☀️";
}

export default function PredictionPanel({ prediction, errorMessage, riskWindow }) {
  if (errorMessage) {
    return (
      <section className="prediction-panel error">
        <h2>예측 결과</h2>
        <p>{errorMessage}</p>
      </section>
    );
  }

  if (!prediction) {
    return (
      <section className="prediction-panel loading">
        <h2>예측 결과</h2>
        <p>날짜를 선택하면 예측 방문객 수와 혼잡도를 보여드립니다.</p>
      </section>
    );
  }

  return (
    <section className="prediction-panel">
      <div className="panel-heading">
        <div>
          <p className="panel-kicker">선택한 날짜</p>
          <h2>{prediction.date}</h2>
        </div>
        <div className="panel-metric">
          <span>예상 방문객 수</span>
          <strong>{prediction.predictedVisitors.toLocaleString("ko-KR")}명</strong>
        </div>
      </div>
      <div className="panel-badges">
        <span className="badge primary">{prediction.congestionLevel}</span>
        {["forecast", "mid_forecast"].includes(prediction.weatherSource) ? (
          <span className="badge weather">실제 날씨값 반영</span>
        ) : null}
        {prediction.seasonFlags.map((season) => (
          <span key={season} className="badge">
            {season}
          </span>
        ))}
      </div>
      <p className="date-comment">{prediction.dateComment}</p>
      <p className="random-comment" style={{ color: prediction.textColor }}>
        {prediction.randomComment}
      </p>
      <dl className="detail-grid">
        <div>
          <dt>쉬는 날 느낌 🗓️</dt>
          <dd>{getHolidayLabel(prediction)}</dd>
        </div>
        <div>
          <dt>하늘 분위기 🌤️</dt>
          <dd>{getSimpleWeatherLabel(prediction.averagePrecipitation)}</dd>
        </div>
      </dl>
      {riskWindow?.days?.length ? (
        <div className="risk-window-summary">
          <h3>선택일 주변 5일 분위기</h3>
          <div className="risk-strip">
            {riskWindow.days.map((item) => (
              <span
                key={item.date}
                className="risk-chip"
                style={{ backgroundColor: hexToRgba(item.riskColor, 0.5) }}
                title={`${item.date} · ${item.predictedVisitors.toLocaleString("ko-KR")}명`}
              >
                {item.date.slice(5)}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}
