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
    <main>
      <h1>Algae RAG Platform</h1>
      <p>백엔드 상태: {health}</p>
    </main>
  )
}

export default App