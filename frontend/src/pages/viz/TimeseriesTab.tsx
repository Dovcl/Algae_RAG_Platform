import { useState } from 'react'
import { useSearchParams } from 'react-router'
import { fetchLocations, fetchTimeseries } from '../../api.ts'
import LocationSelect from '../../components/LocationSelect.tsx'
import MockBadge from '../../components/MockBadge.tsx'
import StageBadge from '../../components/StageBadge.tsx'
import { FIRST_BASE_DATE, LAST_BASE_DATE } from '../../lib/chat.ts'
import { formatCells } from '../../lib/stages.ts'
import { useFetch } from '../../lib/useFetch.ts'
import TimeseriesChart, { type Visible } from './TimeseriesChart.tsx'

const RANGES = [
  { weeks: 8, label: '최근 8주' },
  { weeks: 12, label: '최근 12주' },
  { weeks: 52, label: '최근 1년' },
]

const TOGGLES: { key: keyof Visible; label: string }[] = [
  { key: 'observed', label: '관측값' },
  { key: 'median', label: '예측 중앙값' },
  { key: 'band', label: '예측 범위' },
]

/** 기준일의 다음 주. 차트는 예측 대상 주까지 보여 준다. */
function nextWeek(date: string): string {
  const next = new Date(`${date}T00:00:00Z`)
  next.setUTCDate(next.getUTCDate() + 7)
  return next.toISOString().slice(0, 10)
}

function TimeseriesTab() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [visible, setVisible] = useState<Visible>({ observed: true, median: true, band: true })

  const locations = useFetch('locations', fetchLocations)
  const firstCode = locations.status === 'ok' ? locations.data[0]?.code : undefined
  const code = searchParams.get('location') ?? firstCode ?? null
  const weeks = Number(searchParams.get('weeks')) || 52
  const dateParam = searchParams.get('date')
  const date = dateParam && !Number.isNaN(Date.parse(dateParam)) ? dateParam : null
  const end = date && nextWeek(date)

  const series = useFetch(code === null ? null : `${code}-${weeks}-${end}`, () =>
    fetchTimeseries(code ?? '', weeks, end),
  )

  function setParam(name: string, value: string) {
    const next = new URLSearchParams(searchParams)
    if (value) next.set(name, value)
    else next.delete(name)
    setSearchParams(next, { replace: true })
  }

  if (locations.status === 'loading') return <p className="text-sm text-slate-500">불러오는 중…</p>
  if (locations.status === 'error')
    return (
      <p className="text-sm text-stage-4">지점 목록을 불러오지 못했습니다. ({locations.error})</p>
    )

  // 기준일을 정하면 답변의 숫자와 맞춘다: 관측은 기준일, 예측은 그다음 주.
  // 기준일 뒤의 관측은 그 시점에 알 수 없던 값이라 그리지 않는다.
  const raw = series.status === 'ok' ? series.data.points : []
  const points = date
    ? raw.map((point) => (point.date > date ? { ...point, observed: null } : point))
    : raw
  const current = date ? points.findLast((point) => point.date <= date) : points.at(-1)
  const forecast = date ? points.find((point) => point.date > date) : points.at(-1)

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-200 bg-white p-3">
        <LocationSelect
          locations={locations.data}
          value={code ?? ''}
          onChange={(value) => setParam('location', value)}
        />

        <label className="flex items-center gap-1.5 text-xs text-slate-700">
          기준일
          <input
            type="date"
            value={date ?? ''}
            min={FIRST_BASE_DATE}
            max={LAST_BASE_DATE}
            onChange={(e) => setParam('date', e.target.value)}
            className="h-9 rounded border border-slate-300 bg-white px-2 text-sm"
          />
        </label>

        <div className="inline-flex gap-1 rounded bg-slate-100 p-1">
          {RANGES.map((range) => (
            <button
              key={range.weeks}
              type="button"
              onClick={() => setParam('weeks', String(range.weeks))}
              className={`rounded px-3 py-1 text-xs ${
                weeks === range.weeks
                  ? 'bg-white font-semibold text-primary shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {range.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-700">
          {TOGGLES.map((toggle) => (
            <label key={toggle.key} className="flex items-center gap-1.5">
              <input
                type="checkbox"
                checked={visible[toggle.key]}
                onChange={(e) => setVisible({ ...visible, [toggle.key]: e.target.checked })}
              />
              {toggle.label}
            </label>
          ))}
        </div>

        {series.status === 'ok' && series.data.backend === 'mock' && (
          <span className="ml-auto">
            <MockBadge />
          </span>
        )}
      </div>

      {series.status === 'loading' && <p className="text-sm text-slate-500">불러오는 중…</p>}
      {series.status === 'error' && (
        <p className="text-sm text-stage-4">시계열을 불러오지 못했습니다. ({series.error})</p>
      )}

      {series.status === 'ok' && !forecast && (
        <p className="text-sm text-slate-500">이 기준일의 시계열 자료가 없습니다.</p>
      )}

      {series.status === 'ok' && forecast && (
        <>
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-lg border border-observed-border bg-observed-bg p-4">
              <p className="text-xs font-semibold text-observed">
                관측값 · {current?.date ?? date}
              </p>
              <p className="mt-1 text-2xl font-bold text-slate-900">
                {current?.observed == null ? '-' : formatCells(current.observed)}
                <span className="ml-1 text-sm font-normal text-slate-500">cells/mL</span>
              </p>
            </div>
            <div className="rounded-lg border border-forecast-border bg-forecast-bg p-4">
              <p className="text-xs font-semibold text-forecast">예측 중앙값 · {forecast.date}</p>
              <p className="mt-1 text-2xl font-bold text-slate-900">
                {formatCells(forecast.predicted.q50)}
                <span className="ml-1 text-sm font-normal text-slate-500">cells/mL</span>
              </p>
              {forecast.predicted.q20 !== null && forecast.predicted.q80 !== null && (
                <p className="mt-1 text-xs text-slate-600">
                  범위 {formatCells(forecast.predicted.q20)} ~{' '}
                  {formatCells(forecast.predicted.q80)}
                </p>
              )}
            </div>
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <p className="text-xs font-semibold text-slate-500">예측 단계</p>
              <p className="mt-2">
                <StageBadge stage={forecast.stage} />
              </p>
              <p className="mt-2 text-xs text-slate-600">예측 중앙값 기준</p>
            </div>
          </div>

          <TimeseriesChart
            points={points}
            visible={visible}
            fileName={`timeseries_${series.data.location}_${forecast.date}`}
          />
        </>
      )}
    </div>
  )
}

export default TimeseriesTab