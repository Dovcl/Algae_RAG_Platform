type PlaceholderProps = {
  title: string
}

function Placeholder({ title }: PlaceholderProps) {
  return (
    <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-slate-300 bg-white text-sm text-slate-400">
      {title} 화면이 들어갈 자리
    </div>
  )
}

export default Placeholder