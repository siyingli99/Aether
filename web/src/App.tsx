import { useEffect, useMemo, useState } from 'react'
import {
  Activity, Box, Camera, Check, ChevronRight, CirclePlay, Code2, Database,
  Map as MapIcon, Plus, Route, Save, Settings2, Sparkles, Square, Trash2,
} from 'lucide-react'
import { api } from './api'
import CameraPanel from './CameraPanel'
import MapCanvas from './MapCanvas'
import type { ActorEdit, Catalog, Experiment, Job, Preview, RoadPose, SceneDetail } from './types'

const blankPose = (): RoadPose => ({ x: 0, y: 0, z: 0, yaw: 0 })
const newId = () => `studio-${Math.random().toString(36).slice(2, 10)}`

function normalizeExperiment(value: Experiment): Experiment {
  const algorithm = value.algorithm || { adapter: 'uniad', cuda: '0' }
  const runtime = value.runtime || { max_steps: 39, seed: 0, difix: {} as Experiment['runtime']['difix'] }
  const difix = runtime.difix || ({} as Experiment['runtime']['difix'])
  return {
    ...value,
    ego: { ...value.ego, start: { ...blankPose(), ...(value.ego?.start || {}) } },
    algorithm: { ...algorithm, adapter: algorithm.adapter || 'uniad', cuda: algorithm.cuda || '0' },
    runtime: {
      ...runtime,
      max_steps: runtime.max_steps ?? 39,
      seed: runtime.seed ?? 0,
      difix: {
        ...difix,
        enabled: difix.enabled ?? true, cuda: difix.cuda || '0',
        max_side: difix.max_side ?? 1024, model: difix.model || 'nvidia/difix',
        timestep: difix.timestep ?? 199, dtype: difix.dtype || 'bf16',
      },
    },
  }
}

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [detail, setDetail] = useState<SceneDetail | null>(null)
  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [tab, setTab] = useState<'map' | 'cameras'>('map')
  const [frame, setFrame] = useState(0)
  const [drawMode, setDrawMode] = useState(false)
  const [filter, setFilter] = useState<'all' | 'vehicles' | 'people'>('all')
  const [message, setMessage] = useState('Ready')
  const [busy, setBusy] = useState(false)
  const [job, setJob] = useState<Job | null>(null)
  const [logs, setLogs] = useState('')

  useEffect(() => {
    api.catalog().then(value => {
      setCatalog(value)
      const first = value.scenes.find(scene => scene.id === 'nuscenes-0383') || value.scenes[0]
      if (first) void loadScene(first.id)
    }).catch(error => setMessage(error.message))
  }, [])

  async function loadScene(sceneId: string) {
    setBusy(true); setMessage('Loading scene catalog…'); setSelectedId(null); setFrame(0)
    try {
      const value = await api.scene(sceneId)
      setDetail(value)
      setExperiment(normalizeExperiment(value.experiment))
      setMessage(`${value.actors.length} reconstructed actors · ${value.topology.lanes.length} map lanes`)
    } catch (error) { setMessage((error as Error).message) } finally { setBusy(false) }
  }

  useEffect(() => {
    if (!detail || detail.preview.status !== 'running') return
    const timer = window.setInterval(async () => {
      const preview = await api.preview(detail.scene.id)
      setDetail(current => current ? { ...current, preview } : current)
      if (preview.status !== 'running') window.clearInterval(timer)
    }, 1800)
    return () => window.clearInterval(timer)
  }, [detail?.scene.id, detail?.preview.status])

  const selected = experiment?.actors.find(actor => actor.id === selectedId) || null
  const sourceActor = detail?.actors.find(actor => actor.id === selectedId) || null
  const visibleActors = useMemo(() => {
    if (!experiment) return []
    return experiment.actors.filter(actor => filter === 'all' || (filter === 'vehicles' ? actor.node_class === 'RigidNodes' : actor.node_class !== 'RigidNodes'))
  }, [experiment, filter])

  function updateActor(id: string, mutate: (actor: ActorEdit) => ActorEdit) {
    setExperiment(current => current ? { ...current, actors: current.actors.map(actor => actor.id === id ? mutate(actor) : actor) } : current)
  }

  function moveActor(id: string, pose: RoadPose) {
    updateActor(id, actor => {
      if (actor.controller.kind === 'open_loop') {
        const trajectory = actor.controller.trajectory.length >= 2
          ? actor.controller.trajectory.map((point, index) => index === 0 ? { ...point, ...pose } : point)
          : [{ ...pose, t: 0 }, { ...pose, x: pose.x + 8, t: 4 }]
        return { ...actor, operation: actor.source === 'external' ? 'add' : 'edit', start: pose, controller: { ...actor.controller, trajectory } }
      }
      return { ...actor, operation: actor.source === 'external' ? 'add' : 'edit', start: pose }
    })
  }

  function drawPoint(pose: RoadPose) {
    if (!selectedId) return
    updateActor(selectedId, actor => {
      const existing = actor.controller.kind === 'open_loop' ? actor.controller.trajectory : []
      const point = { ...pose, t: existing.length ? existing.at(-1)!.t + 2 : 0 }
      const trajectory = [...existing, point]
      return {
        ...actor, operation: actor.source === 'external' ? 'add' : 'edit', start: trajectory[0],
        controller: { kind: 'open_loop', trajectory, params: {} },
      }
    })
  }

  function addVehicle() {
    if (!catalog || !experiment) return
    const asset = catalog.assets.find(item => item.available)
    if (!asset) { setMessage('No 3DRealCar asset is available'); return }
    const id = newId()
    const actor: ActorEdit = {
      id, source: 'external', operation: 'add', node_class: 'RigidNodes', label: 'vehicle',
      asset_id: asset.id, start: blankPose(), size: [2, 4.5, 1.6],
      controller: { kind: 'open_loop', trajectory: [{ ...blankPose(), t: 0 }, { ...blankPose(), x: 12, t: 6 }], params: {} },
    }
    setExperiment({ ...experiment, actors: [...experiment.actors, actor] })
    setSelectedId(id); setTab('map'); setDrawMode(true)
    setMessage('Vehicle added. Click the map to draw its trajectory.')
  }

  async function generatePreview() {
    if (!detail) return
    const preview = await api.createPreview(detail.scene.id)
    setDetail({ ...detail, preview })
  }

  async function validateAndSave(run = false) {
    if (!experiment) return
    setBusy(true)
    try {
      const validation = await api.validate(experiment)
      const manifest = await api.save(experiment)
      const id = manifest.experiment_id
      setExperiment(current => current ? { ...current, id } : current)
      setMessage(validation.warnings.length ? validation.warnings.join(' · ') : 'Experiment is valid and saved')
      if (run) {
        const started = await api.run(id)
        setJob(started); setLogs(''); setMessage(`Closed loop ${started.id} started`)
        connectLogs(started.id)
      }
    } catch (error) { setMessage((error as Error).message) } finally { setBusy(false) }
  }

  function connectLogs(jobId: string) {
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
    const socket = new WebSocket(`${protocol}://${location.host}/api/jobs/${jobId}/logs`)
    socket.onmessage = event => {
      const value = JSON.parse(event.data)
      if (value.text) setLogs(current => (current + value.text).slice(-18000))
      if (value.status) {
        setMessage(`Closed loop ${value.status}`)
        void api.job(jobId).then(setJob)
      }
    }
  }

  if (!catalog || !detail || !experiment) return <div className="boot"><Sparkles/><span>{message}</span></div>

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><Sparkles size={19}/></div><div><strong>AETHER</strong><span>STUDIO</span></div></div>
        <div className="experiment-title">
          <span>EXPERIMENT</span>
          <input value={experiment.name} onChange={event => setExperiment({ ...experiment, name: event.target.value })} />
        </div>
        <div className="top-actions">
          <button onClick={() => void validateAndSave(false)} disabled={busy}><Check size={15}/> Validate</button>
          <button onClick={() => void validateAndSave(false)} disabled={busy}><Save size={15}/> Save</button>
          <button className="run" onClick={() => void validateAndSave(true)} disabled={busy || job?.status === 'running'}><CirclePlay size={16}/> Run closed loop</button>
        </div>
      </header>

      <aside className="left-panel">
        <section className="panel-section scenes">
          <div className="section-title"><Database size={14}/><span>TRAINED SCENE</span></div>
          {catalog.scenes.map(scene => <button key={scene.id} className={`scene-card ${scene.id === detail.scene.id ? 'active' : ''}`} onClick={() => void loadScene(scene.id)}>
            <div><strong>{scene.name.split(' · ')[0]}</strong><span>{scene.name.split(' · ')[1]}</span></div>
            <div className="badges"><span>{scene.has_map ? 'HD MAP' : 'NO MAP'}</span>{scene.has_limsim && <span>LIMSIM</span>}</div>
          </button>)}
        </section>
        <section className="panel-section actors">
          <div className="section-title"><Box size={14}/><span>SCENE ACTORS</span><em>{experiment.actors.length}</em></div>
          <div className="segmented compact">
            {(['all','vehicles','people'] as const).map(value => <button className={filter === value ? 'active' : ''} onClick={() => setFilter(value)} key={value}>{value}</button>)}
          </div>
          <div className="actor-list">
            {visibleActors.map(actor => <button key={actor.id} className={`actor-row ${selectedId === actor.id ? 'active' : ''} ${actor.operation === 'remove' ? 'removed' : ''}`} onClick={() => setSelectedId(actor.id)}>
              <span className={`actor-dot ${actor.source}`}/><div><strong>{actor.source === 'external' ? 'External vehicle' : `${actor.node_class.replace('Nodes','')} ${actor.instance_id}`}</strong><small>{actor.operation} · {actor.controller.kind}</small></div><ChevronRight size={14}/>
            </button>)}
          </div>
          <button className="add-vehicle" onClick={addVehicle}><Plus size={16}/> Add 3DRealCar vehicle</button>
        </section>
      </aside>

      <main className="workspace">
        <div className="workspace-bar">
          <div className="tabs"><button className={tab === 'map' ? 'active' : ''} onClick={() => setTab('map')}><MapIcon size={15}/> Road editor</button><button className={tab === 'cameras' ? 'active' : ''} onClick={() => setTab('cameras')}><Camera size={15}/> Six cameras</button></div>
          <div className="canvas-tools"><button className={drawMode ? 'active' : ''} onClick={() => setDrawMode(!drawMode)}><Route size={15}/> Draw trajectory</button><span className="coordinate-pill">+X forward · +Y left · yaw CCW</span></div>
        </div>
        <div className="canvas-wrap">
          {tab === 'map' ? <MapCanvas topology={detail.topology} actors={detail.actors} edits={experiment.actors} selectedId={selectedId} frame={frame} drawMode={drawMode} onSelect={setSelectedId} onMove={moveActor} onDrawPoint={drawPoint}/> : <CameraPanel preview={detail.preview} sceneId={detail.scene.id} onGenerate={() => void generatePreview()}/>}
        </div>
        <div className="timeline">
          <Square size={11} fill="currentColor"/><span>FRAME</span><strong>{String(frame).padStart(3,'0')}</strong><input type="range" min={0} max={150} value={frame} onChange={event => setFrame(Number(event.target.value))}/><span>15.0 s</span>
        </div>
        <div className="statusbar"><Activity size={13}/><span>{message}</span>{job && <span className={`job-pill ${job.status}`}>JOB {job.id} · {job.status}</span>}</div>
      </main>

      <aside className="right-panel">
        <div className="inspector-title"><Settings2 size={15}/><span>INSPECTOR</span></div>
        {selected ? <ActorInspector actor={selected} sourceActor={sourceActor} assets={catalog.assets} controllers={catalog.controllers} hasLimsim={detail.scene.has_limsim} onChange={next => updateActor(selected.id, () => next)} onDelete={() => updateActor(selected.id, actor => ({ ...actor, operation: 'remove' }))}/> : <ExperimentInspector experiment={experiment} adapters={catalog.adapters} onChange={setExperiment}/>}
      </aside>

      {logs && <section className="run-console"><div><Code2 size={14}/><strong>RUN LOG</strong><button onClick={() => setLogs('')}>×</button></div><pre>{logs}</pre></section>}
    </div>
  )
}

function ActorInspector({ actor, sourceActor, assets, controllers, hasLimsim, onChange, onDelete }: {
  actor: ActorEdit; sourceActor: SceneDetail['actors'][number] | null; assets: Catalog['assets']; controllers: Catalog['controllers']; hasLimsim: boolean
  onChange: (actor: ActorEdit) => void; onDelete: () => void
}) {
  const isRigid = actor.node_class === 'RigidNodes'
  const operations = actor.source === 'external'
    ? ['add', 'remove']
    : isRigid ? ['keep','remove','edit','replace'] : ['keep','remove']
  const setOperation = (operation: ActorEdit['operation']) => onChange({
    ...actor, operation,
    asset_id: operation === 'replace' ? (actor.asset_id || assets[0]?.id) : actor.asset_id,
  })
  const pose = actor.start || blankPose()
  const updatePose = (key: keyof RoadPose, value: number) => onChange({ ...actor, start: { ...pose, [key]: value } })
  return <div className="inspector-body">
    <div className="entity-heading"><span className={`large-dot ${actor.source}`}/><div><strong>{actor.source === 'external' ? 'External vehicle' : actor.id}</strong><small>{actor.node_class} · {sourceActor ? `frames ${sourceActor.first_frame}–${sourceActor.last_frame}` : 'new asset'}</small></div></div>
    <label className="field"><span>ACTION</span><select value={actor.operation} onChange={event => setOperation(event.target.value as ActorEdit['operation'])}>{operations.map(value => <option key={value}>{value}</option>)}</select></label>
    {(actor.operation === 'replace' || actor.source === 'external') && <label className="field"><span>VEHICLE ASSET</span><select value={actor.asset_id || ''} onChange={event => onChange({ ...actor, asset_id: event.target.value })}>{assets.filter(asset => asset.available).map(asset => <option key={asset.id} value={asset.id}>{asset.name}</option>)}</select><small>Source: 3DRealCar</small></label>}
    {actor.operation !== 'remove' && <>
      <label className="field"><span>CONTROLLER</span><select value={actor.controller.kind} onChange={event => {
        const kind = event.target.value as ActorEdit['controller']['kind']
        let trajectory = actor.controller.trajectory
        if (kind === 'open_loop' && trajectory.length < 2) trajectory = [{ ...pose, t: 0 }, { ...pose, x: pose.x + 10, t: 5 }]
        const plugin = kind === 'plugin' ? (actor.controller.plugin || controllers.find(value => value.kind === 'plugin')?.id || null) : actor.controller.plugin
        onChange({ ...actor, operation: actor.source === 'external' ? 'add' : actor.operation === 'keep' ? 'edit' : actor.operation, controller: { ...actor.controller, kind, trajectory, plugin } })
      }}><option value="recorded" disabled={actor.source === 'external'}>Recorded trajectory</option><option value="open_loop">Open-loop trajectory</option><option value="limsim" disabled={!hasLimsim || actor.source !== 'external'}>LimSim</option><option value="plugin">Python plugin</option></select></label>
      {actor.controller.kind === 'plugin' && <label className="field"><span>REGISTERED PLUGIN</span><select value={actor.controller.plugin || ''} onChange={event => onChange({ ...actor, controller: { ...actor.controller, plugin: event.target.value } })}><option value="" disabled>Select plugin</option>{controllers.filter(value => value.kind === 'plugin').map(value => <option key={value.id} value={value.id}>{value.name}</option>)}</select>{!controllers.some(value => value.kind === 'plugin') && <small>Register a plugin in configs/studio/catalog.yaml.</small>}</label>}
      <div className="field"><span>START POSE · ROAD FRAME</span><div className="number-grid">{(['x','y','yaw'] as const).map(key => <label key={key}><em>{key === 'yaw' ? 'yaw°' : `${key} m`}</em><input type="number" step={key === 'yaw' ? 1 : .25} value={key === 'yaw' ? pose.yaw * 180 / Math.PI : pose[key]} onChange={event => updatePose(key, key === 'yaw' ? Number(event.target.value) * Math.PI / 180 : Number(event.target.value))}/></label>)}</div></div>
      {actor.controller.kind === 'open_loop' && <div className="trajectory-summary"><Route size={15}/><div><strong>{actor.controller.trajectory.length} waypoints</strong><small>Use “Draw trajectory” and click the road map.</small></div><button onClick={() => onChange({ ...actor, controller: { ...actor.controller, trajectory: [] } })}>Clear</button></div>}
    </>}
    <button className="danger" onClick={onDelete}><Trash2 size={15}/> Remove from scene</button>
  </div>
}

function ExperimentInspector({ experiment, adapters, onChange }: { experiment: Experiment; adapters: Catalog['adapters']; onChange: (value: Experiment) => void }) {
  const ego = experiment.ego
  const updateEgo = (key: keyof RoadPose, value: number) => onChange({ ...experiment, ego: { ...ego, start: { ...ego.start, [key]: value } } })
  return <div className="inspector-body">
    <div className="subhead">EGO & ALGORITHM</div>
    <label className="field"><span>E2E ADAPTER</span><select value={experiment.algorithm.adapter} onChange={event => onChange({ ...experiment, algorithm: { ...experiment.algorithm, adapter: event.target.value } })}>{adapters.map(adapter => <option key={adapter.id} value={adapter.id}>{adapter.name}</option>)}</select></label>
    <div className="field"><span>EGO START · ROAD FRAME</span><div className="number-grid">{(['x','y','yaw'] as const).map(key => <label key={key}><em>{key === 'yaw' ? 'yaw°' : `${key} m`}</em><input type="number" value={key === 'yaw' ? ego.start.yaw * 180 / Math.PI : ego.start[key]} onChange={event => updateEgo(key, key === 'yaw' ? Number(event.target.value) * Math.PI / 180 : Number(event.target.value))}/></label>)}</div></div>
    <label className="field"><span>START SPEED · M/S</span><input type="number" min="0" step=".5" value={ego.speed} onChange={event => onChange({ ...experiment, ego: { ...ego, speed: Number(event.target.value) } })}/></label>
    <div className="subhead">RUNTIME</div>
    <label className="field"><span>MAX STEPS</span><input type="number" min="1" value={experiment.runtime.max_steps} onChange={event => onChange({ ...experiment, runtime: { ...experiment.runtime, max_steps: Number(event.target.value) } })}/></label>
    <label className="toggle-row"><div><strong>Difix camera repair</strong><small>Applied to all six inputs before E2E inference.</small></div><input type="checkbox" checked={experiment.runtime.difix.enabled} onChange={event => onChange({ ...experiment, runtime: { ...experiment.runtime, difix: { ...experiment.runtime.difix, enabled: event.target.checked } } })}/></label>
    <label className="field"><span>DIFIX MAX SIDE</span><select value={experiment.runtime.difix.max_side} onChange={event => onChange({ ...experiment, runtime: { ...experiment.runtime, difix: { ...experiment.runtime.difix, max_side: Number(event.target.value) } } })}><option>512</option><option>768</option><option>1024</option></select></label>
    <div className="coordinate-card"><Route size={17}/><div><strong>Canonical road frame</strong><small>+x forward · +y left · +yaw counter-clockwise</small></div></div>
  </div>
}
