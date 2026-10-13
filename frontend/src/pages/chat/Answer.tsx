import { ChartLine, Info } from 'lucide-react'
import { Link } from 'react-router'
import type { ChatResponse } from '../../api.ts'
import MockBadge from '../../components/MockBadge.tsx'
import StageBadge from '../../components/StageBadge.tsx'
import EvidenceCards from './EvidenceCards.tsx'

type AnswerProps = {
  data: ChatResponse
}

/** 본문의 [1], [2] 같은 출처 번호만 색을 입힌다 */
function renderWithCitations(text: string) {
  return text.split(/(\[\d+\])/).map((part, index) =>
    /^\[\d+\]$/.test(part) ? (
      <span key={index} className="text-xs font-semibold text-guideline">
        {part}
      </span>
    ) : (
      part
    ),
  )
}

function Answer({ data }: AnswerProps) {
  const isMock = data.llm === 'mock' || data.predict_backend === 'mock'

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex items-center gap-2">
        {data.location && (
          <span className="text-sm font-semibold text-slate-900">{data.location}</span>
        )}
        {data.risk_level !== null && <StageBadge stage={data.risk_level} />}
        {isMock && <MockBadge />}
      </div>

      <EvidenceCards data={data} />

      <p className="text-[15px] leading-7 text-slate-800">{renderWithCitations(data.answer)}</p>

      {data.management_actions.length > 0 && (
        <div>
          <p className="text-sm font-semibold text-slate-900">권고 조치</p>
          <ol className="mt-2 flex flex-col gap-1.5">
            {data.management_actions.map((action, index) => (
              <li key={action} className="flex items-start gap-2 text-sm text-slate-800">
                <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-primary text-[11px] font-semibold text-white">
                  {index + 1}
                </span>
                {action}
              </li>
            ))}
          </ol>
        </div>
      )}

      {data.limitations && (
        <p className="flex items-start gap-2 rounded bg-slate-50 p-3 text-xs leading-5 text-slate-600">
          <Info className="mt-0.5 size-3.5 shrink-0" />
          {data.limitations}
        </p>
      )}

      {data.evidence.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-500">출처</p>
          <ul className="mt-2 divide-y divide-slate-200 rounded border border-slate-200">
            {data.evidence.map((item, index) => (
              <li key={item.id} className="flex items-baseline gap-3 px-3 py-2 text-sm">
                <span className="text-xs font-semibold text-guideline">[{index + 1}]</span>
                <span className="min-w-0 flex-1 text-slate-800">{item.document}</span>
                <span className="text-xs text-slate-500">
                  {item.publisher} · p.{item.page} · {item.section}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {data.location && (
        <div>
          <Link
            to={`/viz/timeseries?location=${data.location}`}
            className="inline-flex h-8 items-center gap-1.5 rounded border border-slate-300 px-3 text-xs font-semibold text-slate-700 hover:bg-slate-50"
          >
            <ChartLine className="size-3.5" />
            시계열에서 보기
          </Link>
        </div>
      )}
    </div>
  )
}

export default Answer
