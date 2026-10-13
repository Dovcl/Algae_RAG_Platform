import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  LineElement,
  LogarithmicScale,
  PointElement,
  Tooltip,
  type ChartData,
  type ChartOptions,
  type Plugin,
} from 'chart.js'
import { Download } from 'lucide-react'
import { useRef } from 'react'
import { Line } from 'react-chartjs-2'
import type { TimeseriesPoint } from '../../api.ts'
import { STAGE_LINES, formatCells } from '../../lib/stages.ts'

ChartJS.register(CategoryScale, LogarithmicScale, LineElement, PointElement, Filler, Tooltip)

const OBSERVED = '#1d4ed8'
const FORECAST = '#6d28d9'
const BAND = 'rgba(109, 40, 217, 0.15)'

// 로그 눈금은 0을 그리지 못한다. 1 미만은 바닥(1)에 붙여 그리고 말풍선에는 원래 값을 보여 준다.
const floor = (value: number | null) => (value === null ? null : Math.max(value, 1))

// 단계 경계에 점선과 이름표를 그린다
const stageLines: Plugin<'line'> = {
  id: 'stageLines',
  afterDatasetsDraw(chart) {
    const { ctx, chartArea, scales } = chart
    for (const { stage, value, color } of STAGE_LINES) {
      const y = scales.y.getPixelForValue(value)
      if (y < chartArea.top || y > chartArea.bottom) continue
      ctx.save()
      ctx.strokeStyle = color
      ctx.fillStyle = color
      ctx.lineWidth = 1
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(chartArea.left, y)
      ctx.lineTo(chartArea.right, y)
      ctx.stroke()
      ctx.font = '11px sans-serif'
      ctx.textAlign = 'right'
      ctx.fillText(`${stage}단계 이상`, chartArea.right - 4, y - 4)
      ctx.restore()
    }
  },
}

// PNG로 저장할 때 배경이 투명하지 않도록 흰색을 먼저 칠한다
const whiteBackground: Plugin<'line'> = {
  id: 'whiteBackground',
  beforeDraw(chart) {
    const { ctx, width, height } = chart
    ctx.save()
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, width, height)
    ctx.restore()
  },
}

export type Visible = {
  observed: boolean
  median: boolean
  band: boolean
}

type TimeseriesChartProps = {
  points: TimeseriesPoint[]
  visible: Visible
  fileName: string
}

function TimeseriesChart({ points, visible, fileName }: TimeseriesChartProps) {
  const chartRef = useRef<ChartJS<'line'>>(null)

  const data: ChartData<'line', (number | null)[]> = {
    labels: points.map((p) => p.date.slice(5).replace('-', '/')),
    datasets: [
      {
        label: '범위 하한',
        data: points.map((p) => floor(p.predicted.q20)),
        hidden: !visible.band,
        borderWidth: 0,
        pointRadius: 0,
      },
      {
        label: '범위 상한',
        data: points.map((p) => floor(p.predicted.q80)),
        hidden: !visible.band,
        borderWidth: 0,
        pointRadius: 0,
        fill: '-1',
        backgroundColor: BAND,
      },
      {
        label: '예측 중앙값',
        data: points.map((p) => floor(p.predicted.q50)),
        hidden: !visible.median,
        borderColor: FORECAST,
        backgroundColor: FORECAST,
        borderWidth: 2,
        borderDash: [6, 4],
        pointRadius: 2,
        pointStyle: 'rectRot',
      },
      {
        label: '관측값',
        data: points.map((p) => floor(p.observed)),
        hidden: !visible.observed,
        borderColor: OBSERVED,
        backgroundColor: OBSERVED,
        borderWidth: 2,
        pointRadius: 2,
      },
    ],
  }

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      tooltip: {
        callbacks: {
          title: (items) => points[items[0].dataIndex].date,
          label: (item) => {
            const point = points[item.dataIndex]
            const original = [
              point.predicted.q20,
              point.predicted.q80,
              point.predicted.q50,
              point.observed,
            ][item.datasetIndex]
            if (original === null) return ''
            return `${item.dataset.label}: ${formatCells(original)} cells/mL`
          },
        },
      },
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: '#64748b', maxTicksLimit: 13 } },
      y: {
        type: 'logarithmic',
        min: 1,
        max: 10_000_000,
        grid: { color: '#e2e8f0' },
        border: { display: false },
        title: { display: true, text: '남조류 세포수 (cells/mL)', color: '#64748b' },
        ticks: {
          color: '#64748b',
          callback: (value) => {
            const n = Number(value)
            return Number.isInteger(Math.log10(n)) ? formatCells(n) : ''
          },
        },
      },
    },
  }

  function downloadPng() {
    const chart = chartRef.current
    if (!chart) return
    const link = document.createElement('a')
    link.href = chart.toBase64Image('image/png', 1)
    link.download = `${fileName}.png`
    link.click()
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-4 text-xs text-slate-600">
          <span className="flex items-center gap-1.5">
            <span className="w-5 border-t-2" style={{ borderColor: OBSERVED }} />
            관측값
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-5 border-t-2 border-dashed" style={{ borderColor: FORECAST }} />
            예측 중앙값
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-4 rounded-sm" style={{ backgroundColor: BAND }} />
            예측 범위 (20~80%)
          </span>
        </div>
        <button
          type="button"
          onClick={downloadPng}
          className="flex h-8 items-center gap-1.5 rounded border border-slate-300 px-3 text-xs font-semibold text-slate-700 hover:bg-slate-50"
        >
          <Download className="size-3.5" />
          PNG
        </button>
      </div>

      <div className="relative mt-3 h-96">
        <Line
          ref={chartRef}
          data={data}
          options={options}
          plugins={[whiteBackground, stageLines]}
          role="img"
          aria-label="주별 남조류 세포수의 관측값과 예측값"
        />
      </div>
    </div>
  )
}

export default TimeseriesChart