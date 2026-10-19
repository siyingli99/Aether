import { Trash2 } from 'lucide-react'
import type { Selection } from './MapEditor'
import type { EditorState } from './scenario'
import type { Catalog, PlacedVehicle, SceneDetail } from './types'
import { Field, Num, Section, Segmented, Toggle, deg, rad } from './ui'

interface Props {
  state: EditorState
  scene: SceneDetail
  catalog: Catalog
  selection: Selection
  egoPose: { x: number; y: number; yaw: number }
  update: (patch: Partial<EditorState>) => void
  updateVehicle: (id: string, patch: Partial<PlacedVehicle>) => void
  removeVehicle: (id: string) => void
  replaceRecorded: (ref: string) => void
}

export default function Inspector(p: Props) {
  const sel = p.selection
  if (sel?.kind === 'vehicle') {
    const v = p.state.placed.find(x => x.id === sel.id)
    if (v) return <VehicleInspector v={v} {...p} />
  }
  if (sel?.kind === 'ego') {
    const pose = p.egoPose
    return (
      <Section title="Ego vehicle" aside={p.state.egoStart &&
        <button className="link" onClick={() => p.update({ egoStart: null })}>reset to logged</button>}>
        <p className="muted">{p.state.egoStart ? 'Custom start pose (box centre).' : 'Starts where the recorded vehicle started.'}
          {' '}Drag on the map; drag the handle to rotate.</p>
        <div className="grid3">
          <Field label="x"><Num value={round(pose.x)} step={0.5} unit="m" onChange={x => p.update({ egoStart: { ...pose, x } })} /></Field>
          <Field label="y"><Num value={round(pose.y)} step={0.5} unit="m" onChange={y => p.update({ egoStart: { ...pose, y } })} /></Field>
          <Field label="heading"><Num value={deg(pose.yaw)} step={5} unit="°" onChange={d => p.update({ egoStart: { ...pose, yaw: rad(d) } })} /></Field>
        </div>
        <Field label="Vehicle model" hint="collision box & kinematics">
          <select value={p.state.vehicle} onChange={e => p.update({ vehicle: e.target.value })}>
            <option value="auto">auto · {p.scene.vehicle.name} (recording vehicle)</option>
            {p.catalog.vehicles.map(v => <option key={v.name} value={v.name}>{v.name} · {v.length.toFixed(2)}×{v.width.toFixed(2)} m</option>)}
          </select>
        </Field>
        <Field label="Initial speed">
          <div className="row">
            <Segmented value={p.state.speedMode} onChange={speedMode => p.update({ speedMode })}
                       options={[{ id: 'logged', label: 'Recorded' }, { id: 'fixed', label: 'Fixed' }]} />
            {p.state.speedMode === 'fixed' && <Num value={p.state.speed} step={0.5} min={0} unit="m/s" onChange={speed => p.update({ speed })} />}
          </div>
        </Field>
      </Section>
    )
  }
  if (sel?.kind === 'recorded') {
    const a = p.scene.recorded.find(r => r.ref === sel.ref)
    if (!a) return null
    const kept = p.state.keep.includes(a.ref)
    return (
      <Section title={`Reconstructed ${a.kind}`}>
        <dl className="facts">
          <dt>Node</dt><dd className="mono">{a.ref}</dd>
          <dt>Size</dt><dd>{a.length.toFixed(2)} × {a.width.toFixed(2)} × {a.height.toFixed(2)} m</dd>
          <dt>Visible</dt><dd>frames {a.first_frame}–{a.last_frame}</dd>
        </dl>
        <div className="stack">
          <Toggle checked={kept} label="Keep on its recorded trajectory"
                  onChange={on => p.update({ keep: on ? [...p.state.keep, a.ref] : p.state.keep.filter(k => k !== a.ref) })} />
          {a.kind === 'vehicle' && <button className="secondary" onClick={() => p.replaceRecorded(a.ref)}>
            Edit: control this car in the scenario</button>}
        </div>
      </Section>
    )
  }
  return (
    <Section title="Selection">
      <p className="muted">Click a vehicle on the map to edit it. Use <b>Add vehicle</b> to place traffic,
        or click a reconstructed (purple) car to keep or re-drive it.</p>
      <ul className="legend">
        <li><i style={{ background: '#ff7a45' }} />Ego</li>
        <li><i style={{ background: '#38bdf8' }} />Placed vehicle (Gaussian asset)</li>
        <li><i style={{ background: '#a78bfa' }} />Reconstructed vehicle kept / re-driven</li>
        <li><i className="dashed" />Reconstructed vehicle hidden</li>
        <li><i style={{ background: '#4f8cff', height: 3 }} />Recorded ego route</li>
      </ul>
    </Section>
  )
}

function VehicleInspector({ v, updateVehicle, removeVehicle }: { v: PlacedVehicle } & Props) {
  const up = (patch: Partial<PlacedVehicle>) => updateVehicle(v.id, patch)
  return (
    <Section title={v.replaces ? `Re-driven ${v.replaces}` : 'Placed vehicle'}
             aside={<button className="icon danger" title="Delete" onClick={() => removeVehicle(v.id)}><Trash2 size={15} /></button>}>
      <Field label="ID"><input value={v.id} onChange={e => up({ id: e.target.value })} /></Field>
      <div className="grid3">
        <Field label="x"><Num value={round(v.x)} step={0.5} unit="m" onChange={x => up({ x })} /></Field>
        <Field label="y"><Num value={round(v.y)} step={0.5} unit="m" onChange={y => up({ y })} /></Field>
        <Field label="heading"><Num value={deg(v.yaw)} step={5} unit="°" onChange={d => up({ yaw: rad(d) })} /></Field>
      </div>
      <Field label="Behaviour">
        <Segmented value={v.controller} onChange={controller => up({ controller })}
                   options={[{ id: 'static', label: 'Parked' }, { id: 'constant', label: 'Constant speed' }, { id: 'plugin', label: 'Python class' }]} />
      </Field>
      {v.controller === 'constant' && <Field label="Speed"><Num value={v.speed} step={0.5} min={0} unit="m/s" onChange={speed => up({ speed })} /></Field>}
      {v.controller === 'plugin' && <Field label="Controller class" hint="package.module:Class (VehicleController)">
        <input className="mono" value={v.pluginClass} onChange={e => up({ pluginClass: e.target.value })} /></Field>}
      {!v.replaces && <p className="muted small">Rendered with the first Gaussian vehicle asset; the collision box uses the asset's size.</p>}
    </Section>
  )
}

const round = (x: number) => Math.round(x * 100) / 100
