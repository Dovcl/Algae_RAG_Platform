// 백엔드 응답의 모양. backend/app/routers/predict.py의 Pydantic 모델과 맞춘다.
export type LocationInfo = {
  code: string
  river: string
  river_name: string
}

export type Quantiles = {
  q20: number
  q50: number
  q80: number
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

export function fetchTimeseries(location: string, weeks: number) {
  const params = new URLSearchParams({ location, weeks: String(weeks) })
  return getJson<TimeseriesResponse>(`/api/timeseries?${params}`)
}