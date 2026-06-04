const HOLIDAY_ICON_RULES = [
  ["설날", "seollal"],
  ["추석", "chuseok"],
  ["어린이", "children"],
  ["부처", "buddha"],
  ["현충", "memorial"],
  ["한글", "hangeul"],
  ["기독", "christmas"],
  ["성탄", "christmas"],
  ["신정", "new-year"],
  ["삼일", "korea"],
  ["광복", "korea"],
  ["개천", "korea"],
  ["선거", "election"],
  ["대체", "substitute"],
  ["임시", "special"],
];

export const HOLIDAY_ICON_PATHS = {
  "new-year": "/holiday-icons/new-year.png",
  seollal: "/holiday-icons/seollal.png",
  korea: "/holiday-icons/korea.png",
  children: "/holiday-icons/children.png",
  buddha: "/holiday-icons/buddha.png",
  memorial: "/holiday-icons/memorial.png",
  chuseok: "/holiday-icons/chuseok.png",
  hangeul: "/holiday-icons/hangeul.png",
  christmas: "/holiday-icons/christmas.png",
  substitute: "/holiday-icons/substitute.png",
  election: "/holiday-icons/election.png",
  special: "/holiday-icons/special.png",
};

export function holidayIconForName(name) {
  const match = HOLIDAY_ICON_RULES.find(([keyword]) => name.includes(keyword));
  return match?.[1] ?? "special";
}

export async function loadHolidaysKrMonth(year, month) {
  const serviceKey = import.meta.env.VITE_HOLIDAYS_KR_SERVICE_KEY;
  if (!serviceKey) {
    return [];
  }

  try {
    const module = await import("holidays-kr");
    const HolidaysKr = module.default?.default ?? module.default ?? module;
    HolidaysKr.serviceKey = serviceKey;
    return await HolidaysKr.getHolidays({ year, month, monthCount: 1 });
  } catch (error) {
    console.warn("holidays-kr lookup failed; using backend holiday data.", error);
    return [];
  }
}

export function mergeHolidaysKrIntoPayload(payload, holidays) {
  if (!payload || !holidays.length) {
    return payload;
  }

  const holidayByDate = new Map();
  holidays.forEach((holiday) => {
    const existing = holidayByDate.get(holiday.dateStr) ?? [];
    existing.push({
      name: holiday.name,
      icon: holidayIconForName(holiday.name),
    });
    holidayByDate.set(holiday.dateStr, existing);
  });

  return {
    ...payload,
    weeks: payload.weeks.map((week) =>
      week.map((day) => {
        if (!day || !holidayByDate.has(day.date)) {
          return day;
        }
        const holidayIcons = holidayByDate.get(day.date);
        return {
          ...day,
          isHoliday: true,
          holidayNames: holidayIcons.map((item) => item.name),
          holidayIcons,
        };
      }),
    ),
  };
}
