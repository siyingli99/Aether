import type { Catalog, PlacedVehicle, Scenario, SceneDetail } from './types'

export type TrafficKind = 'none' | 'limsim' | 'idm' | 'class'

/** Everything the editor lets the user change, in UI-friendly form. */
export interface EditorState {
  sceneId: string
  name: string
  agent: string
  cuda: string
  dt: number
  maxSteps: number
  seed: number
  tracker: 'pure_pursuit' | 'ilqr'
  vehicle: string                 // 'auto' or a profile name
  speedMode: 'logged' | 'fixed'
  speed: number
  egoStart: { x: number; y: number; yaw: number } | null    // null = logged start
  traffic: TrafficKind
  maxVehicles: number
  idmCount: number
  idmSpeed: number
  limsimPort: number
  trafficClass: string            // package.module:Class implementing TrafficModel
  trafficParams: string           // JSON object passed as constructor kwargs
  assets: string[]
  hideOriginals: boolean
  keep: string[]
  placed: PlacedVehicle[]
  refine: boolean
  refineRequired: boolean
  stopOn: string[]
  atFaultOnly: boolean
  offRoadTolerance: number
  bevFps: number
  trajectoryHz: number
}

export const EVENTS = [
  { id: 'vehicle_collision', label: 'Vehicle collision' },
  { id: 'static_collision', label: 'Static collision' },
  { id: 'off_road', label: 'Off road (HD map)' },
  { id: 'off_scene', label: 'Off reconstructed scene' },
]

export function initialState(catalog: Catalog, scene: SceneDetail | null, sceneId: string): EditorState {
  const d = catalog.defaults
  return {
    sceneId, name: '', agent: 'replay', cuda: '0',
    dt: d.sim.dt, maxSteps: d.sim.max_steps, seed: d.sim.seed,
    tracker: d.sim.tracker ?? 'pure_pursuit',
    vehicle: 'auto', speedMode: 'logged', speed: 5, egoStart: null,
    traffic: scene && catalog.scenes.find(s => s.id === sceneId)?.has_network ? 'limsim' : (scene?.lanes.length ? 'idm' : 'none'),
    maxVehicles: 3, idmCount: 3, idmSpeed: 8, limsimPort: 8766,
    trafficClass: 'my_traffic:MyTrafficModel', trafficParams: '{}',
    assets: catalog.assets.slice(0, 1), hideOriginals: true, keep: [], placed: [],
    refine: false, refineRequired: false,
    stopOn: [...d.termination.stop_on], atFaultOnly: false, offRoadTolerance: d.termination.off_road_tolerance,
    bevFps: d.output.bev_fps, trajectoryHz: d.output.trajectory_hz,
  }
}

const CONSTANT = 'aether.traffic.builtin:ConstantVelocity'

/** Constructor kwargs typed by the user; invalid JSON is reported by the form. */
export function parseParams(text: string): Record<string, unknown> {
  try {
    const value = JSON.parse(text || '{}')
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {}
  } catch {
    return {}
  }
}

export function toScenario(s: EditorState): Scenario {
  const model = s.traffic === 'limsim' ? { type: 'limsim', host: '127.0.0.1', port: s.limsimPort }
    : s.traffic === 'idm' ? { type: 'idm', count: s.idmCount, desired_speed: s.idmSpeed }
    : s.traffic === 'class' ? { class: s.trafficClass, params: parseParams(s.trafficParams) } : null
  const vehicles = s.placed.map(v => ({
    id: v.id, size: [v.length, v.width, v.height], start: { x: v.x, y: v.y, yaw: v.yaw },
    controller: v.controller === 'static' ? { type: 'static' }
      : v.controller === 'constant' ? { type: 'plugin', class: CONSTANT, params: { speed_mps: v.speed } }
      : { type: 'plugin', class: v.pluginClass, params: {} },
    ...(v.replaces ? { render: { original: v.replaces } } : {}),
  }))
  const needsAssets = model !== null || s.placed.some(v => !v.replaces)
  return {
    scene: s.sceneId, name: s.name || null,
    sim: { dt: s.dt, max_steps: s.maxSteps, seed: s.seed, tracker: s.tracker },
    ego: { vehicle: s.vehicle, start: s.egoStart ?? 'logged', speed: s.speedMode === 'logged' ? 'logged' : s.speed, allow_reverse: false },
    traffic: { hide_originals: s.hideOriginals, keep: s.keep, assets: needsAssets ? s.assets : [],
               max_vehicles: s.maxVehicles, model, vehicles },
    refine: { enabled: s.refine, required: s.refine && s.refineRequired, model: 'nvidia/difix', max_side: 1024,
              timestep: 199, dtype: 'bf16', cuda: null },
    termination: { stop_on: s.stopOn, at_fault_only: s.atFaultOnly, off_road_tolerance: s.offRoadTolerance },
    output: { videos: true, bev_fps: s.bevFps, trajectory_hz: s.trajectoryHz },
    agent: { name: s.agent, cuda: s.cuda, options: {} },
  }
}

/** Minimal YAML writer for the scenario export (plain data only). */
export function toYaml(value: unknown, indent = 0): string {
  const pad = '  '.repeat(indent)
  const scalar = (v: unknown) => v === null ? 'null' : typeof v === 'string'
    ? (/^[\w./:-]+$/.test(v) && !/^(true|false|null|yes|no|on|off|\d.*)$/i.test(v) ? v : JSON.stringify(v))
    : typeof v === 'number' ? String(Math.round(v * 1e4) / 1e4) : String(v)
  if (Array.isArray(value)) {
    if (!value.length) return '[]'
    if (value.every(v => typeof v !== 'object' || v === null)) return `[${value.map(scalar).join(', ')}]`
    return '\n' + value.map(v => `${pad}- ${toYaml(v, indent + 1).trimStart()}`).join('\n')
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).filter(([, v]) => v !== undefined)
    if (!entries.length) return '{}'
    return (indent ? '\n' : '') + entries.map(([k, v]) => {
      const inner = toYaml(v, indent + 1)
      return `${pad}${k}:${inner.startsWith('\n') ? inner : ' ' + inner}`
    }).join('\n')
  }
  return scalar(value)
}

export function nearestLaneHeading(scene: SceneDetail, x: number, y: number): number | null {
  let best = Infinity, heading: number | null = null
  for (const lane of scene.lanes) {
    const c = lane.centerline
    for (let i = 0; i + 1 < c.length; i++) {
      const [ax, ay] = c[i], [bx, by] = c[i + 1]
      const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy || 1e-9
      const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2))
      const d = Math.hypot(x - ax - t * dx, y - ay - t * dy)
      if (d < best) { best = d; heading = Math.atan2(dy, dx) }
    }
  }
  return best < 6 ? heading : null
}
