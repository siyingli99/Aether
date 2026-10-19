import { useEffect, useMemo, useState } from 'react'
import { BarChart3, Car, Crosshair, Layers, Map as MapIcon, MousePointer2, Navigation, Play, Plus, Radio, Scan } from 'lucide-react'
import { api } from './api'
import Inspector from './Inspector'
import LiveView from './LiveView'
import MapEditor, { type Selection, type Tool } from './MapEditor'
import Results from './Results'
import Settings from './Settings'
import { initialState, nearestLaneHeading, toScenario, type EditorState } from './scenario'
import type { Catalog, Job, PlacedVehicle, SceneDetail } from './types'

type View = 'scenario' | 'live' | 'results'
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [scene, setScene] = useState<SceneDetail | null>(null)
  const [state, setState] = useState<EditorState | null>(null)
  const [view, setView] = useState<View>('scenario')
  const [tool, setTool] = useState<Tool>('select')
  const [selection, setSelection] = useState<Selection>(null)
  const [layers, setLayers] = useState({ lanes: true, route: true, recorded: true, labels: true })
  const [fit, setFit] = useState(0)
  const [job, setJob] = useState<Job | null>(null)
  const [focusRun, setFocusRun] = useState<string | null>(null)
  const [notice, setNotice] = useState<{ kind: 'info' | 'error'; text: string } | null>(null)

  useEffect(() => {
    api.catalog().then(c => {
      setCatalog(c)
      const first = c.scenes.find(s => s.id === 'nuscenes-0383') ?? c.scenes.find(s => !s.error)
      if (first) void openScene(c, first.id)
    }).catch(e => setNotice({ kind: 'error', text: `Backend not reachable: ${e.message}` }))
    api.jobs().then(js => { const running = js.find(j => j.status === 'running'); if (running) { setJob(running); setView('live') } })
  }, [])

  async function openScene(c: Catalog, id: string) {
    try {
      const detail = await api.scene(id)
      setScene(detail)
      setState(prev => ({ ...initialState(c, detail, id), agent: prev?.agent ?? 'replay', cuda: prev?.cuda ?? '0' }))
      setSelection(null)
    } catch (e) { setNotice({ kind: 'error', text: (e as Error).message }) }
  }

  const update = (patch: Partial<EditorState>) => setState(s => s && { ...s, ...patch })
  const updateVehicle = (id: string, patch: Partial<PlacedVehicle>) => {
    setState(s => s && { ...s, placed: s.placed.map(v => v.id === id ? { ...v, ...patch } : v) })
    if (patch.id && selection?.kind === 'vehicle' && selection.id === id) setSelection({ kind: 'vehicle', id: patch.id })
  }

  const egoPose = useMemo(() => state?.egoStart ?? scene?.ego_start ?? { x: 0, y: 0, yaw: 0 }, [state?.egoStart, scene])
  const sceneInfo = catalog?.scenes.find(s => s.id === state?.sceneId)
  const agent = catalog?.agents.find(a => a.id === state?.agent)

  const addVehicle = (x: number, y: number) => {
    if (!state || !scene) return
    const n = state.placed.length + 1
    let id = `v${n}`
    while (state.placed.some(v => v.id === id)) id += "'"
    const yaw = nearestLaneHeading(scene, x, y) ?? egoPose.yaw
    update({ placed: [...state.placed, { id, x, y, yaw, length: 4.5, width: 1.85, height: 1.5, controller: 'constant', speed: 6, pluginClass: '' }] })
    setSelection({ kind: 'vehicle', id }); setTool('select')
  }

  const replaceRecorded = (ref: string) => {
    const a = scene?.recorded.find(r => r.ref === ref)
    if (!a || !state) return
    const id = `edit-${ref.split('/')[1]}`
    update({ placed: [...state.placed, { id, x: a.x, y: a.y, yaw: a.yaw, length: a.length, width: a.width, height: a.height,
                                        controller: 'static', speed: 0, pluginClass: '', replaces: ref }],
             keep: state.keep.filter(k => k !== ref) })
    setSelection({ kind: 'vehicle', id })
  }

  async function run() {
    if (!state) return
    try {
      const scenario = toScenario(state)
      const check = await api.validate(scenario)
      if (!check.valid) { setNotice({ kind: 'error', text: check.errors?.map(e => `${e.loc.join('.')}: ${e.msg}`).join('; ') ?? 'invalid' }); return }
      const started = await api.run(scenario, state.agent, state.cuda)
      setJob(started); setView('live'); setNotice(null)
    } catch (e) { setNotice({ kind: 'error', text: (e as Error).message }) }
  }

  if (!catalog || !state || !scene) {
    return <div className="boot"><div className="logo">Æ</div><p>{notice?.text ?? 'Connecting to AETHER…'}</p></div>
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand"><div className="logo">Æ</div><div><strong>AETHER</strong><span>Studio</span></div></div>
        <nav className="views">
          <button className={view === 'scenario' ? 'active' : ''} onClick={() => setView('scenario')}><MapIcon size={15} />Scenario</button>
          <button className={view === 'live' ? 'active' : ''} onClick={() => setView('live')}><Radio size={15} />Live{job?.status === 'running' && <i className="pulse" />}</button>
          <button className={view === 'results' ? 'active' : ''} onClick={() => setView('results')}><BarChart3 size={15} />Results</button>
        </nav>
        <div className="run-controls">
          <label className="select-pill"><span>Scene</span>
            <select value={state.sceneId} onChange={e => openScene(catalog, e.target.value)}>
              {catalog.scenes.map(s => <option key={s.id} value={s.id} disabled={!!s.error}>{s.name ?? s.id}</option>)}
            </select></label>
          <label className="select-pill"><span>Planner</span>
            <select value={state.agent} onChange={e => update({ agent: e.target.value })}>
              {catalog.agents.map(a => <option key={a.id} value={a.id}>{a.title}</option>)}
            </select></label>
          <label className="select-pill narrow"><span>GPU</span>
            <input value={state.cuda} onChange={e => update({ cuda: e.target.value })} /></label>
          <button className="primary" disabled={job?.status === 'running'} onClick={run}><Play size={15} />Run</button>
        </div>
      </header>
      {notice && <div className={`notice ${notice.kind}`} onClick={() => setNotice(null)}>{notice.text}</div>}

      {view === 'scenario' && <div className="workspace">
        <aside className="left">
          <div className="scene-card">
            <span className="eyebrow">{scene.dataset} · {(scene.frame_dt * 1000).toFixed(0)} ms frames</span>
            <h2>{scene.name}</h2>
            <dl className="facts">
              <dt>Duration</dt><dd>{scene.duration_s.toFixed(1)} s</dd>
              <dt>Recording car</dt><dd>{scene.vehicle.name}</dd>
              <dt>HD map</dt><dd>{scene.lanes.length ? `${scene.lanes.length} lanes` : 'none'}</dd>
              <dt>Reconstructed</dt><dd>{plural(scene.recorded.filter(r => r.kind === 'vehicle').length, 'vehicle')} · {plural(scene.recorded.filter(r => r.kind !== 'vehicle').length, 'person', 'people')}</dd>
            </dl>
          </div>
          {agent && <div className="agent-card">
            <span className="eyebrow">Planner</span>
            <h3>{agent.title}</h3>
            <p>{agent.description}</p>
            <dl className="facts">
              <dt>Cameras</dt><dd>{agent.cameras.length || 'none'}</dd>
              <dt>LiDAR</dt><dd>{agent.lidar ? 'yes' : 'no'}</dd>
              {agent.plan && <><dt>Plan</dt><dd>{agent.plan.frame} · {agent.plan.axes} · {agent.plan.dt}s</dd></>}
            </dl>
          </div>}
          <div className="placed-list">
            <span className="eyebrow">Placed vehicles ({state.placed.length})</span>
            {state.placed.map(v => <button key={v.id} className={selection?.kind === 'vehicle' && selection.id === v.id ? 'active' : ''}
              onClick={() => setSelection({ kind: 'vehicle', id: v.id })}><Car size={14} />{v.id}<em>{v.controller === 'constant' ? `${v.speed} m/s` : v.controller}</em></button>)}
            {!state.placed.length && <p className="muted small">None — background traffic only.</p>}
          </div>
        </aside>
        <main className="canvas">
          <div className="toolbar">
            <button className={tool === 'select' ? 'active' : ''} onClick={() => setTool('select')} title="Select / move (V)"><MousePointer2 size={16} /></button>
            <button className={tool === 'add' ? 'active' : ''} onClick={() => setTool('add')} title="Add vehicle (A)"><Plus size={16} /><span>Add vehicle</span></button>
            <button className={tool === 'ego' ? 'active' : ''} onClick={() => setTool('ego')} title="Place ego start"><Navigation size={16} /><span>Ego start</span></button>
            <i className="sep" />
            <button onClick={() => setFit(f => f + 1)} title="Fit to route"><Scan size={16} /></button>
            <div className="layer-menu"><Layers size={16} />
              {(['lanes', 'route', 'recorded', 'labels'] as const).map(k => (
                <label key={k}><input type="checkbox" checked={layers[k]} onChange={e => setLayers({ ...layers, [k]: e.target.checked })} />{k}</label>))}
            </div>
          </div>
          <MapEditor scene={scene} placed={state.placed} egoStart={egoPose} keep={state.keep} hideOriginals={state.hideOriginals}
                     tool={tool} selection={selection} layers={layers} fitToken={fit}
                     onSelect={setSelection} onAdd={addVehicle} onMoveVehicle={updateVehicle}
                     onMoveEgo={pose => update({ egoStart: pose })} />
          {tool !== 'select' && <div className="tool-hint"><Crosshair size={14} />{tool === 'add' ? 'Click on a lane to place a vehicle (heading follows the lane)' : 'Click to set the ego start'}</div>}
        </main>
        <aside className="right">
          <Inspector state={state} scene={scene} catalog={catalog} selection={selection} egoPose={egoPose}
                     update={update} updateVehicle={updateVehicle} replaceRecorded={replaceRecorded}
                     removeVehicle={id => { update({ placed: state.placed.filter(v => v.id !== id) }); setSelection(null) }} />
          <Settings state={state} catalog={catalog} sceneInfo={sceneInfo} update={update} />
        </aside>
      </div>}
      {view === 'live' && <LiveView job={job} onFinished={j => { setJob(j); if (j.run) setFocusRun(j.run.split('/').pop() ?? null) }} />}
      {view === 'results' && <Results focus={focusRun} />}
    </div>
  )
}
