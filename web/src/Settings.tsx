import { Download } from 'lucide-react'
import { EVENTS, toScenario, toYaml, type EditorState, type TrafficKind } from './scenario'
import type { Catalog, SceneSummary } from './types'
import { Field, Num, Section, Segmented, Toggle } from './ui'

interface Props {
  state: EditorState
  catalog: Catalog
  sceneInfo?: SceneSummary
  update: (patch: Partial<EditorState>) => void
}

export default function Settings({ state: s, catalog, sceneInfo, update }: Props) {
  const exportYaml = () => {
    const text = `# AETHER scenario exported from Studio\n# run: python -m aether run <this file> --agent ${s.agent}\n` + toYaml(toScenario(s)) + '\n'
    const url = URL.createObjectURL(new Blob([text], { type: 'text/yaml' }))
    Object.assign(document.createElement('a'), { href: url, download: `${s.name || s.sceneId}.yaml` }).click()
    URL.revokeObjectURL(url)
  }
  const trafficOptions: { id: TrafficKind; label: string }[] = [
    { id: 'none', label: 'None' }, { id: 'idm', label: 'IDM' },
    ...(sceneInfo?.has_network ? [{ id: 'limsim' as TrafficKind, label: 'LimSim' }] : []),
    { id: 'class', label: 'Python class' },
  ]
  let paramsError = ''
  try { JSON.parse(s.trafficParams || '{}') } catch { paramsError = 'not valid JSON' }
  return (
    <>
      <Section title="Simulation">
        <div className="grid3">
          <Field label="Step"><Num value={s.dt} step={0.05} min={0.05} unit="s" onChange={dt => update({ dt })} /></Field>
          <Field label="Max steps"><Num value={s.maxSteps} step={1} min={1} onChange={maxSteps => update({ maxSteps: Math.round(maxSteps) })} /></Field>
          <Field label="Seed"><Num value={s.seed} step={1} onChange={seed => update({ seed: Math.round(seed) })} /></Field>
        </div>
        <p className="muted small">Episode ≤ {(s.dt * s.maxSteps).toFixed(1)} s; it also ends with the reconstruction.</p>
      </Section>

      <Section title="Background traffic">
        <Segmented value={s.traffic} options={trafficOptions} onChange={traffic => update({ traffic })} />
        {s.traffic === 'idm' && <div className="grid2">
          <Field label="Vehicles"><Num value={s.idmCount} min={0} onChange={v => update({ idmCount: Math.round(v) })} /></Field>
          <Field label="Desired speed"><Num value={s.idmSpeed} step={0.5} unit="m/s" onChange={idmSpeed => update({ idmSpeed })} /></Field>
        </div>}
        {s.traffic === 'limsim' && <Field label="LimSim port" hint="started automatically">
          <Num value={s.limsimPort} onChange={v => update({ limsimPort: Math.round(v) })} /></Field>}
        {s.traffic === 'class' && <>
          <Field label="Model class" hint="package.module:Class, see README › Traffic models">
            <input className="mono" value={s.trafficClass} spellCheck={false}
                   onChange={e => update({ trafficClass: e.target.value })} /></Field>
          <Field label="Parameters" hint={paramsError || 'JSON, passed to the constructor'}>
            <textarea className={`mono ${paramsError ? 'invalid' : ''}`} rows={3} value={s.trafficParams} spellCheck={false}
                      onChange={e => update({ trafficParams: e.target.value })} /></Field>
        </>}
        {s.traffic !== 'none' && <Field label="Max rendered vehicles"><Num value={s.maxVehicles} min={0} onChange={v => update({ maxVehicles: Math.round(v) })} /></Field>}
        <Field label="Vehicle assets" hint="3DRealCar Gaussians, cycled">
          <div className="chips">
            {catalog.assets.length === 0 && <span className="muted small">No assets in data/assets/vehicles</span>}
            {catalog.assets.map(a => {
              const on = s.assets.includes(a)
              return <button key={a} type="button" className={`chip ${on ? 'on' : ''}`}
                             onClick={() => update({ assets: on ? s.assets.filter(x => x !== a) : [...s.assets, a] })}>
                {a.replace('3drealcar_', '')}</button>
            })}
          </div>
        </Field>
        <Toggle checked={s.hideOriginals} label="Hide reconstructed traffic" onChange={hideOriginals => update({ hideOriginals })} />
      </Section>

      <Section title="Termination & metrics">
        <div className="stack">
          {EVENTS.map(e => <Toggle key={e.id} checked={s.stopOn.includes(e.id)} label={`Stop on ${e.label.toLowerCase()}`}
            onChange={on => update({ stopOn: on ? [...s.stopOn, e.id] : s.stopOn.filter(x => x !== e.id) })} />)}
          <Toggle checked={s.atFaultOnly} label="Only at-fault collisions stop the run" onChange={atFaultOnly => update({ atFaultOnly })} />
        </div>
        <Field label="Off-road tolerance"><Num value={s.offRoadTolerance} step={0.1} unit="m" onChange={offRoadTolerance => update({ offRoadTolerance })} /></Field>
      </Section>

      <Section title="Sensors & outputs">
        <div className="stack">
          <Toggle checked={s.refine} label="Difix refinement of camera images" onChange={refine => update({ refine })} />
          {s.refine && <Toggle checked={s.refineRequired} label="Fail the run if refinement fails" onChange={refineRequired => update({ refineRequired })} />}
        </div>
        <div className="grid2">
          <Field label="BEV video"><Num value={s.bevFps} min={1} max={60} unit="fps" onChange={v => update({ bevFps: Math.round(v) })} /></Field>
          <Field label="Trajectory CSV"><Num value={s.trajectoryHz} min={1} unit="Hz" onChange={trajectoryHz => update({ trajectoryHz })} /></Field>
        </div>
        <p className="muted small">Camera resolution is fixed by the scene rig (nuScenes 800×450). UniAD/VAD upsample it to their 1600×900 training input internally.</p>
      </Section>

      <div className="export">
        <Field label="Scenario name"><input value={s.name} placeholder={s.sceneId} onChange={e => update({ name: e.target.value })} /></Field>
        <button className="secondary" onClick={exportYaml}><Download size={15} />Export YAML</button>
      </div>
    </>
  )
}
