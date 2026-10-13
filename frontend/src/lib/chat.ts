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

let lastTurnId = 0

/** 앱이 떠 있는 동안 겹치지 않는 번호. "새 대화"로 목록을 비워도 이어서 센다. */
export function newTurnId(): number {
  lastTurnId += 1
  return lastTurnId
}
