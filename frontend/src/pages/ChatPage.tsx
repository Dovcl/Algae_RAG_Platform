import { ArrowUp } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useOutletContext } from 'react-router'
import { postChat } from '../api.ts'
import { newTurnId, type ChatContext, type Turn } from '../lib/chat.ts'
import Answer from './chat/Answer.tsx'

// mock은 질문에서 지점 코드와 날짜만 읽는다. 그래서 예시에 둘 다 넣었다.
const SUGGESTIONS = [
  'HCB 지점의 2023-08-07 기준 다음 주 전망과 대응을 알려줘',
  'CGB 지점의 2023-07-03 기준 녹조 전망은?',
  'GJB 지점의 현재 남조류 상태를 알려줘',
  'SCB 지점의 2023-09-04 기준 필요한 관리 조치는?',
]

function ChatPage() {
  const { turns, setTurns } = useOutletContext<ChatContext>()
  const [draft, setDraft] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  const pending = turns.some((turn) => turn.result.status === 'loading')

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [turns])

  function send(question: string) {
    const text = question.trim()
    if (!text || pending) return

    const id = newTurnId()
    setTurns((prev) => [...prev, { id, question: text, result: { status: 'loading' } }])
    setDraft('')

    const finish = (result: Turn['result']) =>
      setTurns((prev) => prev.map((turn) => (turn.id === id ? { ...turn, result } : turn)))

    postChat(text)
      .then((data) => finish({ status: 'ok', data }))
      .catch((e: unknown) =>
        finish({ status: 'error', error: e instanceof Error ? e.message : String(e) }),
      )
  }

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-5 px-6 py-6">
          {turns.length === 0 && (
            <div className="py-16 text-center">
              <h1 className="text-xl font-semibold text-slate-900">무엇을 알아볼까요?</h1>
              <p className="mt-2 text-sm text-slate-500">
                지점의 현재 상태, 다음 주 전망, 필요한 관리 조치를 물어보세요.
              </p>
            </div>
          )}

          {turns.map((turn) => (
            <div key={turn.id} className="flex flex-col gap-3">
              <p className="max-w-[80%] self-end rounded-lg bg-primary px-4 py-2 text-sm text-white">
                {turn.question}
              </p>
              {turn.result.status === 'loading' && (
                <p className="text-sm text-slate-500">답변을 작성하는 중…</p>
              )}
              {turn.result.status === 'error' && (
                <div className="flex items-center gap-3 rounded-lg border border-stage-4-border bg-stage-4-bg p-3 text-sm text-stage-4">
                  답변을 받지 못했습니다. ({turn.result.error})
                  <button
                    type="button"
                    onClick={() => send(turn.question)}
                    className="rounded border border-stage-4-border bg-white px-2 py-1 text-xs font-semibold"
                  >
                    다시 시도
                  </button>
                </div>
              )}
              {turn.result.status === 'ok' && <Answer data={turn.result.data} />}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="shrink-0 border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-3xl px-6 py-3">
          <div className="flex flex-wrap gap-2">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                disabled={pending}
                onClick={() => send(suggestion)}
                className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-600 hover:bg-slate-100 disabled:opacity-50"
              >
                {suggestion}
              </button>
            ))}
          </div>

          <form
            className="mt-3 flex gap-2"
            onSubmit={(e) => {
              e.preventDefault()
              send(draft)
            }}
          >
            <input
              aria-label="질문"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="예: HCB 지점의 2023-08-07 기준 전망은?"
              maxLength={2000}
              className="h-10 min-w-0 flex-1 rounded border border-slate-300 px-3 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
            />
            <button
              type="submit"
              aria-label="보내기"
              disabled={pending || draft.trim() === ''}
              className="flex size-10 items-center justify-center rounded bg-primary text-white hover:bg-primary-hover disabled:bg-slate-400"
            >
              <ArrowUp className="size-4" />
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default ChatPage
