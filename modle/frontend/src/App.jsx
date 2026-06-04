import { useEffect, useState } from "react";

import { fetchHealth, fetchMonth, fetchPrediction, fetchRiskWindow } from "./api/client";
import CalendarGrid from "./components/CalendarGrid";
import PredictionPanel from "./components/PredictionPanel";
import RiskLegend from "./components/RiskLegend";
import { loadHolidaysKrMonth, mergeHolidaysKrIntoPayload } from "./utils/holidaysKr";

function toYearMonth(isoDate) {
  const [year, month] = isoDate.split("-").map(Number);
  return { year, month };
}

function shiftMonth(year, month, offset) {
  const current = new Date(year, month - 1 + offset, 1);
  return { year: current.getFullYear(), month: current.getMonth() + 1 };
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [viewYear, setViewYear] = useState(null);
  const [viewMonth, setViewMonth] = useState(null);
  const [riskMode, setRiskMode] = useState(false);
  const [monthPayload, setMonthPayload] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [riskWindow, setRiskWindow] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    fetchHealth()
      .then((payload) => {
        setHealth(payload);
        const today = payload?.today ?? null;
        if (today) {
          setSelectedDate(today);
          const { year, month } = toYearMonth(today);
          setViewYear(year);
          setViewMonth(month);
        }
      })
      .catch((error) => setErrorMessage(error.message));
  }, []);

  useEffect(() => {
    if (!viewYear || !viewMonth) {
      return;
    }
    fetchMonth(viewYear, viewMonth, selectedDate, riskMode)
      .then(async (payload) => {
        const holidaysKrItems = await loadHolidaysKrMonth(viewYear, viewMonth);
        setMonthPayload(mergeHolidaysKrIntoPayload(payload, holidaysKrItems));
      })
      .catch((error) => setErrorMessage(error.message));
  }, [viewYear, viewMonth, selectedDate, riskMode]);

  useEffect(() => {
    if (!selectedDate) {
      return;
    }
    fetchPrediction(selectedDate)
      .then((payload) => {
        setPrediction(payload);
        setErrorMessage("");
      })
      .catch((error) => {
        setPrediction(null);
        setErrorMessage(error.message);
      });
  }, [selectedDate]);

  useEffect(() => {
    if (!selectedDate || !riskMode) {
      setRiskWindow(null);
      return;
    }
    fetchRiskWindow(selectedDate)
      .then((payload) => setRiskWindow(payload))
      .catch(() => setRiskWindow(null));
  }, [selectedDate, riskMode]);

  const handleSelectDate = (day) => {
    if (!day.selectable) {
      setErrorMessage(day.reason || "선택할 수 없는 날짜입니다.");
      return;
    }
    setSelectedDate(day.date);
    setErrorMessage("");
  };

  const handleMonthShift = (offset) => {
    const next = shiftMonth(viewYear, viewMonth, offset);
    setViewYear(next.year);
    setViewMonth(next.month);
  };

  return (
    <main className="app-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">서울대공원 방문자 예측 서비스</p>
          <h1>서울대공원 눈치싸움</h1>
          <p className="hero-description">
            날짜를 아래 달력에서 고르고, 눈치게임에서 이기세요!
          </p>
        </div>
        <div className="status-card">
          <p className="panel-kicker">서비스 상태</p>
          <p>머신러닝 모델을 통해 서울대공원 인원수를 예측합니다. 실제 날씨는 당일 기준 5일까지 반영되요.</p>
        </div>
      </section>

      <section className="calendar-section">
        <div className="calendar-card">
          <div className="calendar-toolbar">
            <div>
              <p className="panel-kicker">달력에서 날짜 선택</p>
              <h2>
                {monthPayload ? `${monthPayload.year}년 ${monthPayload.monthLabel}` : "달력 불러오는 중"}
              </h2>
            </div>
            <div className="calendar-actions">
              <button type="button" className="nav-button" onClick={() => handleMonthShift(-1)}>
                이전 달
              </button>
              <button type="button" className="nav-button" onClick={() => handleMonthShift(1)}>
                다음 달
              </button>
            </div>
          </div>
          <RiskLegend active={riskMode} onToggle={() => setRiskMode((current) => !current)} />
          {monthPayload ? <CalendarGrid weeks={monthPayload.weeks} onSelectDate={handleSelectDate} /> : null}
        </div>
      </section>

      <PredictionPanel prediction={prediction} errorMessage={errorMessage} riskWindow={riskWindow} />
    </main>
  );
}
