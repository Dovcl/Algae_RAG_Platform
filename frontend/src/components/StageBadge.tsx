type StageBadgeProps = {
  stage: number
}

function StageBadge({ stage }: StageBadgeProps) {
  return (
    <span className={`rounded border px-2 py-0.5 text-xs font-semibold stage-${stage}`}>
      {stage}단계
    </span>
  )
}

export default StageBadge