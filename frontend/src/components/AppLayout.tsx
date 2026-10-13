import { Droplets } from 'lucide-react'
import { Outlet } from 'react-router'
import BackendStatus from './BackendStatus.tsx'
import Sidebar from './Sidebar.tsx'

function AppLayout() {
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
        <Sidebar />
        <main className="min-w-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default AppLayout