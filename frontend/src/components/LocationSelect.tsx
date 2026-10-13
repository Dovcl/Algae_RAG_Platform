import type { LocationInfo } from '../api.ts'

type LocationSelectProps = {
  locations: LocationInfo[]
  value: string
  onChange: (code: string) => void
}

function LocationSelect({ locations, value, onChange }: LocationSelectProps) {
  const rivers = [...new Set(locations.map((loc) => loc.river_name))]

  return (
    <select
      aria-label="지점"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="h-9 rounded border border-slate-300 bg-white px-2 text-sm"
    >
      {rivers.map((river) => (
        <optgroup key={river} label={river}>
          {locations
            .filter((loc) => loc.river_name === river)
            .map((loc) => (
              <option key={loc.code} value={loc.code}>
                {loc.code}
              </option>
            ))}
        </optgroup>
      ))}
    </select>
  )
}

export default LocationSelect