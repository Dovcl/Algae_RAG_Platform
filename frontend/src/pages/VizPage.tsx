import { ChartLine, ChartScatter, Map as MapIcon } from 'lucide-react'
import { NavLink, Outlet } from 'react-router'

const tabs = [
  { to: 'map', label: '녹조 예측 지도', icon: MapIcon },
  { to: 'timeseries', label: '시계열 차트', icon: ChartLine },
  { to: 'detail', label: '상세 플롯', icon: ChartScatter },
]

function VizPage() {
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold text-slate-900">시각화</h1>

      <div className="mt-4 inline-flex gap-1 rounded-lg bg-slate-100 p-1">
        {tabs.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-2 rounded-md px-3 py-1.5 text-sm ${
                isActive
                  ? 'bg-white font-semibold text-primary shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`
            }
          >
            <Icon className="size-4" />
            {label}
          </NavLink>
        ))}
      </div>

      <div className="mt-4">
        <Outlet />
      </div>
    </div>
  )
}

export default VizPage