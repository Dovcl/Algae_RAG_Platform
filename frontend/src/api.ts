// 백엔드 응답의 모양. backend/app/routers/predict.py의 Pydantic 모델과 맞춘다.
export type LocationInfo = {
  code: string
  river: string
  river_name: string
  name: string
  zone: string | null
}

export type Quantiles = {
  q20: number | null
  q50: number
  q80: number | null
}

export type TimeseriesPoint = {
  date: string
  observed: number | null
  predicted: Quantiles
  stage: number
}

export type TimeseriesResponse = {
  location: string
  backend: string
  points: TimeseriesPoint[]
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export function fetchLocations() {
  return getJson<LocationInfo[]>('/api/locations')
}

export function fetchTimeseries(location: string, weeks: number, end: string | null) {
  const params = new URLSearchParams({ location, weeks: String(weeks) })
  if (end) params.set('end', end)
  return getJson<TimeseriesResponse>(`/api/timeseries?${params}`)
}

export type CurrentStatus = {
  date: string
  cells_per_ml: number
}

export type ForecastStatus = {
  target_date: string
  input_until: string
  cells_per_ml: Quantiles
  stage: number
}

export type Evidence = {
  id: string
  document: string
  publisher: string
  page: number
  section: string
}

// 필드 이름은 실험 설계안의 출력 형식을 따른다 (DEV_PLAN.md 13장)
export type ChatResponse = {
  question: string
  location: string | null
  current_status: CurrentStatus | null
  forecast_status: ForecastStatus | null
  risk_level: number | null
  management_actions: string[]
  evidence: Evidence[]
  limitations: string | null
  answer: string
  llm: string
  predict_backend: string
}

/** 연속 질문을 위해 함께 보내는 앞 대화 한 턴 */
export type PastTurn = {
  question: string
  answer: string
  location: string | null
  base_date: string | null
}

export async function postChat(
  question: string,
  location: string,
  baseDate: string,
  history: PastTurn[],
): Promise<ChatResponse> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      location: location || null,
      base_date: baseDate || null,
      history,
    }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<ChatResponse>
}
