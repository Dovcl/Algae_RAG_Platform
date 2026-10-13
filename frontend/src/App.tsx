import { useEffect, useState } from 'react'

type HealthResponse = {
  status: string
}

function App() {
  const [health, setHealth] = useState<string>('확인 중...')

  useEffect(() => {
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json() as Promise<HealthResponse>
      })
      .then((data) => setHealth(data.status))
      .catch(() => setHealth('연결 실패'))
  }, [])

  return (
    <main className="p-8">
      <h1 className="text-2xl font-semibold text-primary">Algae RAG Platform</h1>
      <p className="mt-2 text-slate-600">백엔드 상태: {health}</p>
      <div className="mt-4 flex gap-2">
        {[0, 1, 2, 3, 4].map((stage) => (
          <span
            key={stage}
            className={`rounded border px-2 py-1 text-xs font-semibold stage-${stage}`}
          >
            {stage}단계
          </span>
        ))}
        <span className="rounded border border-observed-border bg-observed-bg px-2 py-1 text-xs font-semibold text-observed">
          관측
        </span>
        <span className="rounded border border-forecast-border bg-forecast-bg px-2 py-1 text-xs font-semibold text-forecast">
          예측
        </span>
        <span className="rounded border border-guideline-border bg-guideline-bg px-2 py-1 text-xs font-semibold text-guideline">
          규정
        </span>
      </div>
    </main>
  )
}

export default App