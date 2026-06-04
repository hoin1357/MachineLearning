import { HOLIDAY_ICON_PATHS } from "../utils/holidaysKr";

const WEEKDAY_LABELS = ["일", "월", "화", "수", "목", "금", "토"];

function hexToRgba(hex, alpha) {
  const normalized = hex.replace("#", "");
  const red = parseInt(normalized.slice(0, 2), 16);
  const green = parseInt(normalized.slice(2, 4), 16);
  const blue = parseInt(normalized.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

export default function CalendarGrid({ weeks, onSelectDate }) {
  return (
    <div className="calendar-grid-wrap">
      <div className="weekday-row" role="row">
        {WEEKDAY_LABELS.map((label, index) => (
          <div
            key={label}
            className={`weekday-cell ${index === 0 || index === 6 ? "weekend" : ""}`}
          >
            {label}
          </div>
        ))}
      </div>
      <div className="calendar-grid">
        {weeks.flat().map((day, index) => {
          if (!day) {
            return <div key={`empty-${index}`} className="day-cell empty" aria-hidden="true" />;
          }

          const colorStyle = day.riskColor && !day.isSelected
            ? { backgroundColor: hexToRgba(day.riskColor, 0.5) }
            : undefined;

          return (
            <button
              key={day.date}
              type="button"
              className={[
                "day-cell",
                day.selectable ? "" : "disabled",
                day.isSelected ? "selected" : "",
                day.isToday ? "today" : "",
                day.isWeekend || day.isHoliday ? "holiday" : "",
                day.riskColor ? "risk-active" : "",
              ].join(" ")}
              style={colorStyle}
              onClick={() => onSelectDate(day)}
              aria-disabled={!day.selectable}
              title={day.reason || ""}
            >
              <span className="day-topline">
                <span className="day-number">{day.day}</span>
                {day.holidayIcons?.length ? (
                  <span className="holiday-icon-row" aria-label={day.holidayNames?.join(", ")}>
                    {day.holidayIcons.slice(0, 2).map((holiday) => (
                      <img
                        key={`${day.date}-${holiday.name}`}
                        className="holiday-icon"
                        src={HOLIDAY_ICON_PATHS[holiday.icon] ?? HOLIDAY_ICON_PATHS.special}
                        alt=""
                        title={holiday.name}
                      />
                    ))}
                  </span>
                ) : null}
              </span>
              {day.holidayNames?.length ? (
                <span className="holiday-name">{day.holidayNames[0]}</span>
              ) : null}
              {day.congestionLevel ? <span className="day-risk-label">{day.congestionLevel}</span> : null}
            </button>
          );
        })}
      </div>
    </div>
  );
}
