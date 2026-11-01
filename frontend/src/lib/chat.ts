import type { Dispatch, SetStateAction } from 'react'
import type { ChatResponse } from '../api.ts'
import type { FetchState } from './useFetch.ts'

/** 질문 하나와 그 답. 답은 불러오는 중 / 오류 / 완료 중 하나다. */
export type Turn = {
  id: number
  question: string
  result: FetchState<ChatResponse>
}

/** AppLayout이 들고 있다가 화면에 내려 주는 대화 상태 */
export type ChatContext = {
  turns: Turn[]
  setTurns: Dispatch<SetStateAction<Turn[]>>
}

// 저장 예측이 있는 기준일 범위 (대상 주 2023-01-16 ~ 12-25)
export const FIRST_BASE_DATE = '2023-01-09'
export const LAST_BASE_DATE = '2023-12-18'

/** 답이 다룬 지점과 기준일을 시각화 주소의 검색어로 옮긴다. 지점이 없는 답이면 빈 문자열. */
export function vizSearch(data: ChatResponse): string {
  if (data.location === null) return ''
  const params = new URLSearchParams({ location: data.location })
  // 관측값이 빠진 주에는 current_status가 없어도 예측의 입력 시점이 곧 기준일이다
  const date = data.current_status?.date ?? data.forecast_status?.input_until
  if (date) params.set('date', date)
  return `?${params}`
}

/** 지점을 다룬 가장 최근 답의 vizSearch. 지침만 묻는 질문은 건너뛴다. */
export function lastVizSearch(turns: Turn[]): string {
  for (const { result } of turns.toReversed()) {
    if (result.status !== 'ok') continue
    const search = vizSearch(result.data)
    if (search) return search
  }
  return ''
}

let lastTurnId = 0

/** 앱이 떠 있는 동안 겹치지 않는 번호. "새 대화"로 목록을 비워도 이어서 센다. */
export function newTurnId(): number {
  lastTurnId += 1
  return lastTurnId
}
