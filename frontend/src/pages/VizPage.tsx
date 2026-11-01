import { ChartLine, ChartScatter, Map as MapIcon } from 'lucide-react'
import { Navigate, NavLink, Outlet, useLocation } from 'react-router'

const tabs = [
  { to: 'map', label: '녹조 예측 지도', icon: MapIcon },
  { to: 'timeseries', label: '시계열 차트', icon: ChartLine },
  { to: 'detail', label: '상세 플롯', icon: ChartScatter },
]

/** 질문에서 넘어온 지점·기준일이 있으면 그 값을 쓰는 시계열로, 없으면 지도로 보낸다 */
export function VizIndex() {
  const { search } = useLocation()
  return <Navigate to={{ pathname: search ? 'timeseries' : 'map', search }} replace />
}

function VizPage() {
  const { search } = useLocation()

  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold text-slate-900">시각화</h1>

      <div className="mt-4 inline-flex gap-1 rounded-lg bg-slate-100 p-1">
        {tabs.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={{ pathname: to, search }}
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