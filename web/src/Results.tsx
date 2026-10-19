import { useEffect, useState } from 'react'
import { FileDown, RefreshCw } from 'lucide-react'
import { api } from './api'
import type { Metrics, RunDetail, RunSummary } from './types'

const pct = (v?: number | null) => v == null ? '–' : (v * 100).toFixed(0)
const METRICS: { key: keyof Metrics; label: string; hint: string }[] = [
  { key: 'nc', label: 'NC', hint: 'No at-fault collision' },
  { key: 'dac', label: 'DAC', hint: 'Drivable-area compliance' },
  { key: 'ttc', label: 'TTC', hint: 'Time-to-collision ≥ 0.9 s' },
  { key: 'comfort', label: 'C', hint: 'Comfort bounds' },
  { key: 'ep', label: 'EP', hint: 'Ego progress vs. recorded' },
  { key: 'rc', label: 'RC', hint: 'Route completion' },
]

export default function Results({ focus }: { focus?: string | null }) {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selected, setSelected] = useState<string | null>(focus ?? null)
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [video, setVideo] = useState('overview')

  const refresh = () => api.results().then(r => { setRuns(r); if (!selected && r[0]) setSelected(r[0].id) })
  useEffect(() => { void refresh() }, [])
  useEffect(() => { if (focus) setSelected(focus) }, [focus])
  useEffect(() => { if (selected) api.result(selected).then(setDetail).catch(() => setDetail(null)) }, [selected])

  const m = detail?.metrics
  return (
    <div className="results">
      <aside className="run-list">
        <header><h3>Runs</h3><button className="icon" onClick={refresh} title="Refresh"><RefreshCw size={14} /></button></header>
        <table>
          <thead><tr><th>Run</th><th>Score</th></tr></thead>
          <tbody>{runs.map(r => (
            <tr key={r.id} className={r.id === selected ? 'active' : ''} onClick={() => setSelected(r.id)}>
              <td><b>{r.scene}</b><span>{r.agent} · {r.started?.slice(5, 16).replace('T', ' ')}</span></td>
              <td className="score">{r.metrics ? r.metrics.score.toFixed(2) : <em>{r.complete ? '' : 'incomplete'}</em>}</td>
            </tr>))}
          </tbody>
        </table>
        {!runs.length && <p className="muted pad">No runs yet.</p>}
      </aside>
      {detail && <main className="run-detail">
        <header>
          <div><h2>{detail.scene} <span className="muted">· {detail.agent}</span></h2>
            <p className="muted">{detail.steps} steps · ended by <b>{detail.terminated_by ?? '–'}</b> · {detail.wall_s?.toFixed(0)} s wall time</p></div>
          {m && <div className={`score-big ${m.score < 0.5 ? 'low' : ''}`}><span>PDM-style score</span><b>{m.score.toFixed(3)}</b></div>}
        </header>
        {m && <div className="metric-row">{METRICS.map(x => (
          <div key={x.key} className={`metric ${(m[x.key] as number) <= 0.001 ? 'bad' : (m[x.key] as number) < 0.999 ? 'warn' : ''}`} title={x.hint}>
            <span>{x.label}</span><b>{pct(m[x.key] as number)}</b><em>{x.hint}</em></div>))}
          <div className="metric plain"><span>Distance</span><b>{m.distance_m.toFixed(1)}</b><em>m · {m.mean_speed.toFixed(1)} m/s avg</em></div>
          <div className="metric plain"><span>Min TTC</span><b>{m.min_ttc_s == null ? '∞' : m.min_ttc_s.toFixed(1)}</b><em>seconds</em></div>
        </div>}
        <div className="video-tabs">
          {['overview', 'bev', 'cameras'].map(v => <button key={v} className={video === v ? 'active' : ''} onClick={() => setVideo(v)}>{v}</button>)}
        </div>
        <video key={`${detail.id}-${video}`} className="player" controls autoPlay muted loop src={`${detail.base_url}/videos/${video}.mp4`} />
        <div className="detail-grid">
          <section>
            <h3>Events</h3>
            {detail.events.length ? <table className="events"><thead><tr><th>t (s)</th><th>Type</th><th>Other</th><th>At fault</th></tr></thead>
              <tbody>{detail.events.map((e, i) => <tr key={i}><td>{e.time_s}</td><td>{e.type.replace(/_/g, ' ')}</td><td className="mono">{e.other_id}</td><td>{e.at_fault}</td></tr>)}</tbody></table>
              : <p className="muted">No collisions or violations.</p>}
          </section>
          <section>
            <h3>Data</h3>
            <ul className="files">{detail.files.filter(f => /\.(csv|json)$/.test(f)).map(f => (
              <li key={f}><a href={`${detail.base_url}/${f}`} download><FileDown size={14} />{f}</a></li>))}</ul>
            <p className="muted small">trajectories*.csv: every vehicle, every step — x, y, yaw, speed, acceleration, lane, s/d, global map coordinates.</p>
          </section>
        </div>
      </main>}
    </div>
  )
}
