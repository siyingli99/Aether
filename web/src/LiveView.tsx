import { useEffect, useRef, useState } from 'react'
import { CircleStop, Radio } from 'lucide-react'
import { api } from './api'
import type { Job } from './types'

export default function LiveView({ job, onFinished }: { job: Job | null; onFinished: (job: Job) => void }) {
  const [bev, setBev] = useState<string | null>(null)
  const [cam, setCam] = useState<string | null>(null)
  const [time, setTime] = useState<number | null>(null)
  const [log, setLog] = useState('')
  const [status, setStatus] = useState<Job['status'] | null>(job?.status ?? null)
  const logBox = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (!job) return
    setBev(null); setCam(null); setLog(''); setStatus(job.status)
    const ws = api.liveSocket(job.id)
    ws.onmessage = ev => {
      const msg = JSON.parse(ev.data)
      if (msg.type === 'frame') {
        const url = `data:image/jpeg;base64,${msg.jpeg}`
        msg.kind === 'bev' ? setBev(url) : setCam(url)
        if (typeof msg.time === 'number') setTime(msg.time)
      } else if (msg.type === 'log') {
        setLog(l => (l + msg.text).slice(-60000))
      } else if (msg.type === 'status') {
        setStatus(msg.job.status); onFinished(msg.job)
      }
    }
    return () => ws.close()
  }, [job?.id])

  useEffect(() => { logBox.current?.scrollTo(0, logBox.current.scrollHeight) }, [log])

  if (!job) return <div className="empty"><Radio size={28} /><p>No run in progress. Configure a scenario and press <b>Run</b>.</p></div>
  const steps = [...log.matchAll(/\[step (\d+)\] t=\s*([\d.]+)s v=\s*([\d.]+)/g)]
  const last = steps.at(-1)
  return (
    <div className="live">
      <div className="live-head">
        <div>
          <span className={`badge ${status}`}>{status}</span>
          <strong>{job.scene}</strong><span className="muted"> · {job.agent} · {job.id}</span>
        </div>
        <div className="live-stats">
          <span>t <b>{time?.toFixed(2) ?? '–'} s</b></span>
          <span>step <b>{last?.[1] ?? 0}</b></span>
          <span>v <b>{last?.[3] ?? '–'} m/s</b></span>
          {status === 'running' && <button className="secondary danger" onClick={() => api.cancel(job.id)}><CircleStop size={15} />Stop</button>}
        </div>
      </div>
      <div className="live-frames">
        <figure className="bev">{bev ? <img src={bev} alt="BEV" /> : <div className="placeholder">waiting for the first step…</div>}<figcaption>Bird's-eye view</figcaption></figure>
        <figure className="cams">{cam ? <img src={cam} alt="cameras" /> : <div className="placeholder">loading scene and planner…</div>}<figcaption>Rendered cameras (what the planner sees)</figcaption></figure>
      </div>
      <pre className="console" ref={logBox}>{log || 'starting worker…'}</pre>
    </div>
  )
}
