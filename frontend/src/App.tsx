import { Navigate, Route, Routes } from 'react-router'
import AppLayout from './components/AppLayout.tsx'
import Placeholder from './components/Placeholder.tsx'
import ChatPage from './pages/ChatPage.tsx'
import VizPage from './pages/VizPage.tsx'
import TimeseriesTab from './pages/viz/TimeseriesTab.tsx'

function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to="/chat" replace />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="viz" element={<VizPage />}>
          <Route index element={<Navigate to="map" replace />} />
          <Route path="map" element={<Placeholder title="녹조 예측 지도" />} />
                    <Route path="timeseries" element={<TimeseriesTab />} />
          <Route path="detail" element={<Placeholder title="상세 플롯" />} />
        </Route>
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Route>
    </Routes>
  )
}

export default App