export type RoadPose = { x: number; y: number; z: number; yaw: number }
export type TrajectoryPoint = RoadPose & { t: number; speed?: number | null }
export type Controller = {
  kind: 'recorded' | 'open_loop' | 'limsim' | 'plugin'
  trajectory: TrajectoryPoint[]
  plugin?: string | null
  params: Record<string, unknown>
}
export type ActorEdit = {
  id: string
  source: 'scene' | 'external'
  operation: 'keep' | 'remove' | 'edit' | 'replace' | 'add'
  node_class: 'RigidNodes' | 'DeformableNodes' | 'SMPLNodes'
  instance_id?: number | null
  label: string
  asset_id?: string | null
  start?: RoadPose | null
  size?: [number, number, number] | null
  controller: Controller
}
export type Experiment = {
  schema_version: 'aether.experiment.v1'
  id?: string
  name: string
  scene_id: string
  ego: { start: RoadPose; speed: number }
  actors: ActorEdit[]
  algorithm: { adapter: string; cuda: string; config_path?: string | null }
  runtime: {
    max_steps: number
    seed: number
    difix: { enabled: boolean; cuda: string; max_side: number; model: string; timestep: number; dtype: string }
  }
}
export type SceneEntry = {
  id: string; name: string; available: boolean; has_map: boolean; has_limsim: boolean
}
export type AssetEntry = {
  id: string; name: string; source: string; available: boolean; metadata: Record<string, unknown>
}
export type Catalog = {
  scenes: SceneEntry[]
  assets: AssetEntry[]
  controllers: { id: string; name: string; kind: string; entrypoint?: string }[]
  adapters: { id: string; name: string; available: boolean }[]
  coordinates: Record<string, string>
}
export type SceneActor = {
  id: string; source: 'scene'; node_class: ActorEdit['node_class']; instance_id: number
  label: string; size: [number, number, number]; first_frame: number; last_frame: number
  trajectory: { frame: number; x: number; y: number; yaw: number }[]
}
export type Topology = {
  lanes: { id: string; kind: string; width: number; points: [number, number][] }[]
  route: [number, number][]
  bounds: { min_x: number; max_x: number; min_y: number; max_y: number } | null
  location?: string
}
export type Preview = { status: string; base_url?: string; mosaic?: string; cameras?: Record<string, string> }
export type SceneDetail = {
  scene: SceneEntry
  topology: Topology
  actors: SceneActor[]
  experiment: Experiment
  preview: Preview
}
export type Job = {
  id: string; experiment_id: string; status: string; log_path: string; result_path: string
  pid?: number; returncode?: number; result?: Record<string, unknown>
}
