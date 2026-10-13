import { Droplets } from 'lucide-react'
import { useState } from 'react'
import { Outlet } from 'react-router'
import type { ChatContext, Turn } from '../lib/chat.ts'
import BackendStatus from './BackendStatus.tsx'
import Sidebar from './Sidebar.tsx'

function AppLayout() {
  // 대화는 여기에 둔다. 채팅 화면에 두면 시각화로 갔다 올 때 사라진다.
  const [turns, setTurns] = useState<Turn[]>([])

  return (
    <div className="flex h-screen flex-col">
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-4">
        <div className="flex items-center gap-2">
          <span className="flex size-8 items-center justify-center rounded bg-primary text-white">
            <Droplets className="size-4" />
          </span>
          <span className="text-base font-semibold text-primary">녹조 예측 RAG</span>
        </div>
        <BackendStatus />
      </header>

      <div className="flex min-h-0 flex-1">
        <Sidebar onNewChat={() => setTurns([])} />
        <main className="min-w-0 flex-1 overflow-y-auto">
          <Outlet context={{ turns, setTurns } satisfies ChatContext} />
        </main>
      </div>
    </div>
  )
}

export default AppLayout
