import { BookOpen, Eye, TrendingUp } from 'lucide-react'
import type { ChatResponse } from '../../api.ts'
import { formatCells } from '../../lib/stages.ts'

type EvidenceCardsProps = {
  data: ChatResponse
}

const unavailable = 'mt-2 text-sm text-slate-500'

/** 답의 근거를 종류별로 나눠 보여 준다: 현재 관측 / 미래 예측 / 관리지침 */
function EvidenceCards({ data }: EvidenceCardsProps) {
  const { current_status: current, forecast_status: forecast } = data

  return (
    <div className="grid grid-cols-3 gap-3">
      <div className="rounded-lg border border-observed-border bg-observed-bg p-3">
        <p className="flex items-center gap-1.5 text-xs font-semibold text-observed">
          <Eye className="size-3.5" />
          현재 관측
        </p>
        {current ? (
          <>
            <p className="mt-1 text-xl font-bold text-slate-900">
              {formatCells(current.cells_per_ml)}
              <span className="ml-1 text-xs font-normal text-slate-500">cells/mL</span>
            </p>
            <p className="mt-1 text-xs text-slate-600">{current.date} 관측</p>
          </>
        ) : (
          <p className={unavailable}>관측 정보를 사용할 수 없습니다</p>
        )}
      </div>

      <div className="rounded-lg border border-forecast-border bg-forecast-bg p-3">
        <p className="flex items-center gap-1.5 text-xs font-semibold text-forecast">
          <TrendingUp className="size-3.5" />
          미래 예측
        </p>
        {forecast ? (
          <>
            <p className="mt-1 text-xl font-bold text-slate-900">
              {formatCells(forecast.cells_per_ml.q50)}
              <span className="ml-1 text-xs font-normal text-slate-500">cells/mL</span>
            </p>
            <p className="mt-1 text-xs text-slate-600">
              범위 {formatCells(forecast.cells_per_ml.q20)} ~{' '}
              {formatCells(forecast.cells_per_ml.q80)} · {forecast.target_date} 예측
            </p>
          </>
        ) : (
          <p className={unavailable}>예측 정보를 사용할 수 없습니다</p>
        )}
      </div>

      <div className="rounded-lg border border-guideline-border bg-guideline-bg p-3">
        <p className="flex items-center gap-1.5 text-xs font-semibold text-guideline">
          <BookOpen className="size-3.5" />
          관리지침
        </p>
        {data.evidence.length > 0 ? (
          <>
            <p className="mt-1 text-xl font-bold text-slate-900">
              조치 {data.management_actions.length}건
            </p>
            <p className="mt-1 text-xs text-slate-600">문서 {data.evidence.length}건 참조</p>
          </>
        ) : (
          <p className={unavailable}>관리지침을 사용할 수 없습니다</p>
        )}
      </div>
    </div>
  )
}

export default EvidenceCards
