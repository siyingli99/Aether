import { useMemo, useRef, useState } from 'react'
import { Arrow, Circle, Group, Layer, Line, Rect, Stage, Text } from 'react-konva'
import type Konva from 'konva'
import type { ActorEdit, RoadPose, SceneActor, Topology } from './types'

type Props = {
  topology: Topology
  actors: SceneActor[]
  edits: ActorEdit[]
  selectedId: string | null
  frame: number
  drawMode: boolean
  onSelect: (id: string | null) => void
  onMove: (id: string, pose: RoadPose) => void
  onDrawPoint: (pose: RoadPose) => void
}

const W = 1120
const H = 680
const PAD = 48

export default function MapCanvas({ topology, actors, edits, selectedId, frame, drawMode, onSelect, onMove, onDrawPoint }: Props) {
  const stageRef = useRef<Konva.Stage>(null)
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 })
  const bounds = topology.bounds || { min_x: -50, max_x: 50, min_y: -50, max_y: 50 }
  const baseScale = Math.min((W - PAD * 2) / Math.max(bounds.max_x - bounds.min_x, 1), (H - PAD * 2) / Math.max(bounds.max_y - bounds.min_y, 1))
  const origin = {
    x: PAD - bounds.min_y * baseScale,
    y: H - PAD + bounds.min_x * baseScale,
  }
  const roadToScreen = (x: number, y: number) => ({ x: origin.x + y * baseScale, y: origin.y - x * baseScale })
  const screenToRoad = (sx: number, sy: number): RoadPose => ({ x: (origin.y - sy) / baseScale, y: (sx - origin.x) / baseScale, z: 0, yaw: 0 })
  const editById = useMemo(() => new Map(edits.map(edit => [edit.id, edit])), [edits])

  function poseFor(actor: SceneActor): RoadPose | null {
    const edit = editById.get(actor.id)
    if (edit?.start && edit.operation !== 'keep') return edit.start
    if (edit?.controller.kind === 'open_loop' && edit.controller.trajectory.length) {
      const t = frame * 0.1
      return edit.controller.trajectory.reduce((best, point) => Math.abs(point.t - t) < Math.abs(best.t - t) ? point : best)
    }
    const candidates = actor.trajectory.filter(point => point.frame <= frame)
    const point = candidates.at(-1) || actor.trajectory[0]
    return point ? { x: point.x, y: point.y, z: 0, yaw: point.yaw } : null
  }
  const displayed = [
    ...actors.map(actor => ({ actor, edit: editById.get(actor.id), pose: poseFor(actor) })),
    ...edits.filter(edit => edit.source === 'external').map(edit => ({
      actor: null, edit, pose: edit.start || edit.controller.trajectory[0] || { x: 0, y: 0, z: 0, yaw: 0 },
    })),
  ]

  const pointerRoad = () => {
    const stage = stageRef.current
    const pointer = stage?.getPointerPosition()
    if (!stage || !pointer) return null
    return screenToRoad((pointer.x - view.x) / view.scale, (pointer.y - view.y) / view.scale)
  }

  return (
    <Stage
      ref={stageRef} width={W} height={H} className="map-stage"
      onClick={event => {
        if (event.target === event.target.getStage()) {
          const pose = pointerRoad()
          if (drawMode && pose) onDrawPoint(pose); else onSelect(null)
        }
      }}
      onWheel={event => {
        event.evt.preventDefault()
        const stage = stageRef.current
        const pointer = stage?.getPointerPosition()
        if (!pointer) return
        const factor = event.evt.deltaY > 0 ? 0.9 : 1.1
        const nextScale = Math.min(5, Math.max(0.55, view.scale * factor))
        const roadPointer = { x: (pointer.x - view.x) / view.scale, y: (pointer.y - view.y) / view.scale }
        setView({ scale: nextScale, x: pointer.x - roadPointer.x * nextScale, y: pointer.y - roadPointer.y * nextScale })
      }}
    >
      <Layer x={view.x} y={view.y} scaleX={view.scale} scaleY={view.scale}>
        <Rect width={W} height={H} fill="#0a0f17" />
        {topology.lanes.map(lane => (
          <Line
            key={lane.id}
            points={lane.points.flatMap(([x, y]) => { const p = roadToScreen(x, y); return [p.x, p.y] })}
            stroke={lane.kind === 'lane_connector' ? '#273649' : '#34445a'}
            strokeWidth={lane.width * baseScale * 0.82}
            opacity={0.72}
            lineCap="round" lineJoin="round"
          />
        ))}
        {topology.lanes.map(lane => (
          <Line
            key={`${lane.id}-center`}
            points={lane.points.flatMap(([x, y]) => { const p = roadToScreen(x, y); return [p.x, p.y] })}
            stroke="#71849c" strokeWidth={1 / view.scale} opacity={0.42} dash={[8, 10]}
          />
        ))}
        <Line
          points={topology.route.flatMap(([x, y]) => { const p = roadToScreen(x, y); return [p.x, p.y] })}
          stroke="#7df0c2" strokeWidth={2.5 / view.scale} opacity={0.86} dash={[9, 7]}
        />
        {edits.filter(edit => edit.controller.kind === 'open_loop').map(edit => (
          <Line
            key={`${edit.id}-trajectory`}
            points={edit.controller.trajectory.flatMap(point => { const p = roadToScreen(point.x, point.y); return [p.x, p.y] })}
            stroke={edit.id === selectedId ? '#f6c86e' : '#9c7ae8'}
            strokeWidth={3 / view.scale} dash={[6, 4]} lineCap="round"
          />
        ))}
        {displayed.map(({ actor, edit, pose }) => {
          if (!edit || !pose || edit.operation === 'remove') return null
          const p = roadToScreen(pose.x, pose.y)
          const selected = edit.id === selectedId
          const isVehicle = edit.node_class === 'RigidNodes'
          const color = edit.source === 'external' || edit.operation === 'replace' ? '#f1b96a' : isVehicle ? '#68d7ff' : '#d692ff'
          return (
            <Group
              key={edit.id} x={p.x} y={p.y} rotation={-pose.yaw * 180 / Math.PI + 90}
              draggable={selected}
              onClick={event => { event.cancelBubble = true; onSelect(edit.id) }}
              onDragEnd={event => {
                const next = screenToRoad(event.target.x(), event.target.y())
                next.yaw = pose.yaw
                onMove(edit.id, next)
              }}
            >
              {selected && <Circle radius={17 / view.scale} stroke="#ffffff" strokeWidth={1.5 / view.scale} opacity={0.7} />}
              <Arrow points={[0, 12, 0, -13]} stroke={color} fill={color} pointerLength={7} pointerWidth={8} strokeWidth={6} />
              <Text text={actor ? String(actor.instance_id) : '+'} x={8} y={-7} fill="#dce8f6" fontSize={11 / view.scale} rotation={pose.yaw * 180 / Math.PI - 90} />
            </Group>
          )
        })}
        {(() => { const p = roadToScreen(0, 0); return <Group x={p.x} y={p.y}><Circle radius={5} fill="#7df0c2"/><Text text="EGO" x={9} y={-7} fill="#7df0c2" fontSize={11}/></Group> })()}
      </Layer>
    </Stage>
  )
}
