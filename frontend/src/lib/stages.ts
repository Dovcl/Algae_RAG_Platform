// 단계 경계 (cells/mL). 기준은 DEV_PLAN.md 5장, 색은 index.css의 --color-stage-N과 같다.
// 캔버스는 CSS 변수를 읽지 못해서 차트용으로 색 값을 한 번 더 적는다.
export const STAGE_LINES = [
  { stage: 1, value: 1_000, color: '#16a34a' },
  { stage: 2, value: 10_000, color: '#d97706' },
  { stage: 3, value: 100_000, color: '#ea580c' },
  { stage: 4, value: 1_000_000, color: '#dc2626' },
]

export function formatCells(value: number): string {
  return Math.round(value).toLocaleString('ko-KR')
}