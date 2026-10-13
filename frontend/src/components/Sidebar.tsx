import { ChartLine, MessageSquare, Plus } from 'lucide-react'
import { Link, NavLink } from 'react-router'

const menu = [
  { to: '/chat', label: '채팅', icon: MessageSquare },
  { to: '/viz', label: '시각화', icon: ChartLine },
]

function Sidebar() {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-slate-200 bg-white p-3">
      <Link
        to="/chat"
        className="flex h-10 items-center justify-center gap-2 rounded bg-primary text-sm font-semibold text-white hover:bg-primary-hover"
      >
        <Plus className="size-4" />새 대화
      </Link>

      <nav className="mt-3 flex flex-col gap-1">
        {menu.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-2 rounded px-3 py-2 text-sm ${
                isActive
                  ? 'bg-primary font-semibold text-white'
                  : 'text-slate-600 hover:bg-slate-100'
              }`
            }
          >
            <Icon className="size-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <p className="mt-6 px-3 text-xs font-semibold text-slate-400">대화 기록</p>
      <p className="mt-2 px-3 text-sm text-slate-400">저장된 대화가 없습니다</p>
    </aside>
  )
}

export default Sidebar