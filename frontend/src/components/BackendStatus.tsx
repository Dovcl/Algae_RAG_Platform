import { useEffect, useState } from 'react'

type HealthResponse = {
  status: string
}

type Status = 'checking' | 'ok' | 'error'

const labels: Record<Status, string> = {
  checking: '백엔드 확인 중',
  ok: '백엔드 연결됨',
  error: '백엔드 연결 실패',
}

const dotColors: Record<Status, string> = {
  checking: 'bg-slate-300',
  ok: 'bg-stage-1',
  error: 'bg-stage-4',
}

function BackendStatus() {
  const [status, setStatus] = useState<Status>('checking')

  useEffect(() => {
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json() as Promise<HealthResponse>
      })
      .then((data) => setStatus(data.status === 'ok' ? 'ok' : 'error'))
      .catch(() => setStatus('error'))
  }, [])

  return (
    <span className="flex items-center gap-2 text-xs text-slate-500">
      <span className={`size-2 rounded-full ${dotColors[status]}`} />
      {labels[status]}
    </span>
  )
}

export default BackendStatus