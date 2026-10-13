import { useEffect, useState } from 'react'

type Result<T> = { key: string; data: T } | { key: string; error: string }

export type FetchState<T> =
  | { status: 'loading' }
  | { status: 'error'; error: string }
  | { status: 'ok'; data: T }

/** key가 바뀔 때마다 load()를 다시 부른다. key가 null이면 부르지 않는다. */
export function useFetch<T>(key: string | null, load: () => Promise<T>): FetchState<T> {
  const [result, setResult] = useState<Result<T> | null>(null)

  useEffect(() => {
    if (key === null) return
    let ignore = false
    load()
      .then((data) => {
        if (!ignore) setResult({ key, data })
      })
      .catch((e: unknown) => {
        if (!ignore) setResult({ key, error: e instanceof Error ? e.message : String(e) })
      })
    return () => {
      ignore = true
    }
    // load는 key로 정해지므로 key만 지켜본다
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  if (result === null || result.key !== key) return { status: 'loading' }
  if ('error' in result) return { status: 'error', error: result.error }
  return { status: 'ok', data: result.data }
}