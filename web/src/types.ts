export type XY = [number, number]

export interface SceneSummary {
  id: string; name?: string; dataset?: string; frame_dt?: number; frames?: number
  vehicle?: string; has_map?: boolean; has_network?: boolean; error?: string
}

export interface AgentInfo {
  id: string; title: string; kind: 'builtin' | 'adapter'; description: string
  cameras: string[]; lidar: boolean; plan?: { frame: string; axes: string; dt: number }
}

export interface VehicleProfile { name: string; length: number; width: number; height: number; wheelbase: number }

export interface Catalog {
  scenes: SceneSummary[]; agents: AgentInfo[]; assets: string[]; vehicles: VehicleProfile[]
  defaults: Scenario
}

export interface Lane { id: string; kind: string; centerline: XY[]; polygon: XY[]; width: number }

export interface RecordedActor {
  ref: string; kind: 'vehicle' | 'pedestrian'; x: number; y: number; yaw: number
  length: number; width: number; height: number; first_frame: number; last_frame: number; path: XY[]
}

export interface SceneDetail {
  id: string; name: string; dataset: string; frame_dt: number; duration_s: number
  vehicle: { name: string; length: number; width: number; height: number; wheelbase: number }
  route: XY[]; ego_start: { x: number; y: number; yaw: number }
  lanes: Lane[]; drivable: XY[][]; recorded: RecordedActor[]
}

/** A vehicle placed in the editor (becomes traffic.vehicles[] in the scenario). */
export interface PlacedVehicle {
  id: string; x: number; y: number; yaw: number
  length: number; width: number; height: number
  controller: 'static' | 'constant' | 'plugin'
  speed: number; pluginClass: string
  replaces?: string            // "RigidNodes/3": move a reconstructed car instead of an asset
}

export interface Scenario {
  scene: string; name?: string | null
  sim: { dt: number; max_steps: number; seed: number; tracker?: 'pure_pursuit' | 'ilqr' }
  ego: { vehicle: string; start: 'logged' | { x: number; y: number; yaw: number }; speed: 'logged' | number; allow_reverse: boolean }
  traffic: { hide_originals: boolean; keep: string[]; assets: string[]; max_vehicles: number; model: Record<string, unknown> | null; vehicles: Record<string, unknown>[] }
  refine: { enabled: boolean; required: boolean; model: string; max_side: number; timestep: number; dtype: string; cuda: string | null }
  termination: { stop_on: string[]; at_fault_only: boolean; off_road_tolerance: number; [k: string]: unknown }
  output: { videos: boolean; bev_fps: number; trajectory_hz: number; [k: string]: unknown }
  agent: { name: string | null; cuda: string; options: Record<string, unknown> }
  [k: string]: unknown
}

export interface Job {
  id: string; status: 'running' | 'succeeded' | 'failed' | 'cancelled'
  scene: string; agent: string; created: string; dir: string; run?: string | null; returncode?: number | null
}

export interface Metrics {
  nc: number; dac: number; ttc: number; comfort: number; ep: number; rc: number; score: number
  duration_s: number; distance_m: number; mean_speed: number; collisions: number
  at_fault_collisions: number; static_collisions: number; off_road_steps: number
  min_ttc_s: number | null; terminated_by: string | null; comfort_detail: Record<string, number>
}

export interface RunSummary {
  id: string; scene: string; agent: string; started: string; steps: number
  terminated_by: string | null; wall_s: number; metrics: Metrics | null; complete: boolean
}

export interface RunDetail extends RunSummary {
  events: Record<string, string>[]; files: string[]; base_url: string; run: Record<string, unknown>
}
