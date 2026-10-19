import type { Catalog, Job, RunDetail, RunSummary, Scenario, SceneDetail } from './types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { ...init, headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) } })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail))
  }
  return response.json() as Promise<T>
}

export const api = {
  catalog: () => request<Catalog>('/api/catalog'),
  scene: (id: string) => request<SceneDetail>(`/api/scenes/${encodeURIComponent(id)}`),
  validate: (scenario: Scenario) => request<{ valid: boolean; errors?: { loc: string[]; msg: string }[] }>(
    '/api/validate', { method: 'POST', body: JSON.stringify(scenario) }),
  run: (scenario: Scenario, agent: string, cuda: string) =>
    request<Job>('/api/runs', { method: 'POST', body: JSON.stringify({ scenario, agent, cuda }) }),
  jobs: () => request<Job[]>('/api/jobs'),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  results: () => request<RunSummary[]>('/api/results'),
  result: (id: string) => request<RunDetail>(`/api/results/${encodeURIComponent(id)}`),
  liveSocket: (jobId: string) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    return new WebSocket(`${proto}://${location.host}/api/jobs/${jobId}/live`)
  },
}
