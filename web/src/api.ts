import type { Catalog, Experiment, Job, Preview, SceneDetail } from './types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail))
  }
  return response.json() as Promise<T>
}

export const api = {
  catalog: () => request<Catalog>('/api/catalog'),
  scene: (id: string) => request<SceneDetail>(`/api/scenes/${id}`),
  preview: (id: string) => request<Preview>(`/api/scenes/${id}/preview`),
  createPreview: (id: string) => request<Preview>(`/api/scenes/${id}/preview`, { method: 'POST' }),
  validate: (experiment: Experiment) => request<{ valid: boolean; warnings: string[] }>(
    '/api/experiments/validate', { method: 'POST', body: JSON.stringify(experiment) },
  ),
  save: (experiment: Experiment) => request<Record<string, string>>(
    '/api/experiments', { method: 'POST', body: JSON.stringify(experiment) },
  ),
  run: (id: string) => request<Job>(`/api/experiments/${id}/run`, { method: 'POST' }),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
}
