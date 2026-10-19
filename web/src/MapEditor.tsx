import { useEffect, useMemo, useRef, useState } from 'react'
import { Circle, Group, Layer, Line, Stage, Text } from 'react-konva'
import type Konva from 'konva'
import type { PlacedVehicle, RecordedActor, SceneDetail, XY } from './types'

export type Tool = 'select' | 'add' | 'ego'
export type Selection = { kind: 'vehicle'; id: string } | { kind: 'ego' } | { kind: 'recorded'; ref: string } | null

interface Props {
  scene: SceneDetail
  placed: PlacedVehicle[]
  egoStart: { x: number; y: number; yaw: number }
  keep: string[]
  hideOriginals: boolean
  tool: Tool
  selection: Selection
  layers: { lanes: boolean; route: boolean; recorded: boolean; labels: boolean }
  fitToken: number
  onSelect: (s: Selection) => void
  onAdd: (x: number, y: number) => void
  onMoveVehicle: (id: string, patch: Partial<PlacedVehicle>) => void
  onMoveEgo: (pose: { x: number; y: number; yaw: number }) => void
}

const C = {
  ground: '#0e1117', drivable: '#222935', laneEdge: '#3a4456', center: '#4a5568', route: '#4f8cff',
  ego: '#ff7a45', vehicle: '#38bdf8', selected: '#f8fafc', recorded: '#8b5cf6', kept: '#a78bfa',
  ped: '#f59e0b', text: '#cbd5e1',
}

function corners(x: number, y: number, yaw: number, l: number, w: number): XY[] {
  const c = Math.cos(yaw), s = Math.sin(yaw)
  return [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => {
    const dx = a * l / 2, dy = b * w / 2
    return [x + dx * c - dy * s, y + dx * s + dy * c] as XY
  })
}

type Drag = { kind: 'pan'; sx: number; sy: number; ox: number; oy: number }
  | { kind: 'move'; target: 'vehicle' | 'ego'; id?: string; dx: number; dy: number }
  | { kind: 'rotate'; target: 'vehicle' | 'ego'; id?: string }

export default function MapEditor(p: Props) {
  const box = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 800, h: 600 })
  const [view, setView] = useState({ ox: 400, oy: 300, s: 4 })
  const [drag, setDrag] = useState<Drag | null>(null)
  const [cursor, setCursor] = useState<XY | null>(null)

  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Fit the logged route (plus margin) into view: x up, y left.
  useEffect(() => {
    const pts = p.scene.route.length ? p.scene.route : [[0, 0] as XY]
    const xs = pts.map(q => q[0]), ys = pts.map(q => q[1])
    const [x0, x1, y0, y1] = [Math.min(...xs) - 25, Math.max(...xs) + 25, Math.min(...ys) - 25, Math.max(...ys) + 25]
    const s = Math.min(size.h / (x1 - x0), size.w / (y1 - y0))
    setView({ s, ox: size.w / 2 + ((y0 + y1) / 2) * s, oy: size.h / 2 + ((x0 + x1) / 2) * s })
  }, [p.scene.id, p.fitToken, size.w, size.h])

  const toScreen = (x: number, y: number): XY => [view.ox - y * view.s, view.oy - x * view.s]
  const toWorld = (sx: number, sy: number): XY => [(view.oy - sy) / view.s, (view.ox - sx) / view.s]
  const flat = (pts: XY[]) => pts.flatMap(([x, y]) => toScreen(x, y))

  const pointer = (e: Konva.KonvaEventObject<MouseEvent | WheelEvent>) => {
    const pos = e.target.getStage()!.getPointerPosition()!
    return [pos.x, pos.y] as XY
  }

  const onWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault()
    const [sx, sy] = pointer(e)
    const [wx, wy] = toWorld(sx, sy)
    const s = Math.max(0.5, Math.min(80, view.s * (e.evt.deltaY < 0 ? 1.15 : 1 / 1.15)))
    setView({ s, ox: sx + wy * s, oy: sy + wx * s })
  }

  const onDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (e.target !== e.target.getStage() && e.target.name() !== 'bg') return
    const [sx, sy] = pointer(e)
    if (p.tool === 'add') { const [x, y] = toWorld(sx, sy); p.onAdd(x, y); return }
    if (p.tool === 'ego') {
      const [x, y] = toWorld(sx, sy); p.onMoveEgo({ ...p.egoStart, x, y }); p.onSelect({ kind: 'ego' }); return
    }
    p.onSelect(null)
    setDrag({ kind: 'pan', sx, sy, ox: view.ox, oy: view.oy })
  }

  const onMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const [sx, sy] = pointer(e)
    const [wx, wy] = toWorld(sx, sy)
    setCursor([wx, wy])
    if (!drag) return
    if (drag.kind === 'pan') { setView(v => ({ ...v, ox: drag.ox + sx - drag.sx, oy: drag.oy + sy - drag.sy })); return }
    const target = drag.target === 'ego' ? p.egoStart : p.placed.find(v => v.id === drag.id)
    if (!target) return
    if (drag.kind === 'move') {
      const patch = { x: wx - drag.dx, y: wy - drag.dy }
      drag.target === 'ego' ? p.onMoveEgo({ ...p.egoStart, ...patch }) : p.onMoveVehicle(drag.id!, patch)
    } else {
      const yaw = Math.atan2(wy - target.y, wx - target.x)
      drag.target === 'ego' ? p.onMoveEgo({ ...p.egoStart, yaw }) : p.onMoveVehicle(drag.id!, { yaw })
    }
  }

  const startMove = (target: 'vehicle' | 'ego', id: string | undefined, x: number, y: number) =>
    (e: Konva.KonvaEventObject<MouseEvent>) => {
      e.cancelBubble = true
      const [wx, wy] = toWorld(...pointer(e))
      p.onSelect(target === 'ego' ? { kind: 'ego' } : { kind: 'vehicle', id: id! })
      setDrag({ kind: 'move', target, id, dx: wx - x, dy: wy - y })
    }

  const startRotate = (target: 'vehicle' | 'ego', id?: string) => (e: Konva.KonvaEventObject<MouseEvent>) => {
    e.cancelBubble = true
    setDrag({ kind: 'rotate', target, id })
  }

  const statics = useMemo(() => ({ drivable: p.scene.drivable, lanes: p.scene.lanes }), [p.scene])
  const keep = new Set(p.keep)
  const replaced = new Set(p.placed.map(v => v.replaces).filter(Boolean))
  const ego = p.egoStart
  const L = p.scene.vehicle.length, W = p.scene.vehicle.width

  const car = (x: number, y: number, yaw: number, l: number, w: number, fill: string, stroke: string,
               opts: { dash?: number[]; opacity?: number; strokeWidth?: number } = {}) => (
    <>
      <Line points={flat(corners(x, y, yaw, l, w))} closed fill={fill} stroke={stroke}
            strokeWidth={opts.strokeWidth ?? 1.2} dash={opts.dash} opacity={opts.opacity ?? 1} />
      <Line points={flat(corners(x + 0.3 * l * Math.cos(yaw), y + 0.3 * l * Math.sin(yaw), yaw, 0.16 * l, 0.75 * w))}
            closed fill="rgba(255,255,255,.75)" opacity={opts.opacity ?? 1} listening={false} />
    </>
  )

  const handle = (x: number, y: number, yaw: number, l: number, target: 'vehicle' | 'ego', id?: string) => {
    const [hx, hy] = toScreen(x + (l / 2 + 1.6) * Math.cos(yaw), y + (l / 2 + 1.6) * Math.sin(yaw))
    const [cx, cy] = toScreen(x + l / 2 * Math.cos(yaw), y + l / 2 * Math.sin(yaw))
    return (<>
      <Line points={[cx, cy, hx, hy]} stroke={C.selected} strokeWidth={1} dash={[3, 3]} listening={false} />
      <Circle x={hx} y={hy} radius={6} fill="#0b0d12" stroke={C.selected} strokeWidth={1.5}
              onMouseDown={startRotate(target, id)} />
    </>)
  }

  return (
    <div className={`map-editor tool-${p.tool}`} ref={box}>
      <Stage width={size.w} height={size.h} onWheel={onWheel} onMouseDown={onDown}
             onMouseMove={onMove} onMouseUp={() => setDrag(null)} onMouseLeave={() => { setDrag(null); setCursor(null) }}>
        <Layer listening={false}>
          {statics.drivable.map((ring, i) => <Line key={i} points={flat(ring)} closed fill={C.drivable} />)}
          {p.layers.lanes && statics.lanes.map(lane => (
            <Group key={lane.id}>
              <Line points={flat(lane.polygon)} closed stroke={C.laneEdge} strokeWidth={1} />
              <Line points={flat(lane.centerline)} stroke={C.center} strokeWidth={1} dash={[6, 6]} />
            </Group>
          ))}
          {p.layers.route && <Line points={flat(p.scene.route)} stroke={C.route} strokeWidth={2} opacity={0.75} />}
        </Layer>
        <Layer>
          {p.layers.recorded && p.scene.recorded.map((a: RecordedActor) => {
            const kept = keep.has(a.ref), hidden = p.hideOriginals && !kept, moved = replaced.has(a.ref)
            const sel = p.selection?.kind === 'recorded' && p.selection.ref === a.ref
            if (moved) return null
            return (
              <Group key={a.ref} onMouseDown={e => { e.cancelBubble = true; p.onSelect({ kind: 'recorded', ref: a.ref }) }}>
                {kept && a.path.length > 1 && <Line points={flat(a.path)} stroke={C.kept} strokeWidth={1} dash={[2, 4]} listening={false} />}
                {a.kind === 'pedestrian'
                  ? <Circle x={toScreen(a.x, a.y)[0]} y={toScreen(a.x, a.y)[1]} radius={Math.max(3, 0.4 * view.s)}
                            fill={C.ped} opacity={hidden ? 0.3 : 0.9} stroke={sel ? C.selected : undefined} />
                  : car(a.x, a.y, a.yaw, a.length, a.width, hidden ? 'transparent' : C.kept, sel ? C.selected : C.recorded,
                        { dash: hidden ? [4, 3] : undefined, opacity: hidden ? 0.7 : 0.85, strokeWidth: sel ? 2 : 1 })}
              </Group>
            )
          })}
          {p.placed.map(v => {
            const sel = p.selection?.kind === 'vehicle' && p.selection.id === v.id
            const [tx, ty] = toScreen(v.x, v.y)
            return (
              <Group key={v.id}>
                <Group onMouseDown={startMove('vehicle', v.id, v.x, v.y)}>
                  {car(v.x, v.y, v.yaw, v.length, v.width, v.replaces ? C.kept : C.vehicle, sel ? C.selected : '#0b0d12', { strokeWidth: sel ? 2 : 1 })}
                </Group>
                {p.layers.labels && <Text x={tx + 8} y={ty - 6} text={v.id} fontSize={11} fill={C.text} listening={false} />}
                {sel && handle(v.x, v.y, v.yaw, v.length, 'vehicle', v.id)}
              </Group>
            )
          })}
          <Group onMouseDown={startMove('ego', undefined, ego.x, ego.y)}>
            {car(ego.x, ego.y, ego.yaw, L, W, C.ego, p.selection?.kind === 'ego' ? C.selected : '#0b0d12', { strokeWidth: 2 })}
          </Group>
          {p.layers.labels && <Text x={toScreen(ego.x, ego.y)[0] + 10} y={toScreen(ego.x, ego.y)[1] - 6} text="EGO" fontSize={11} fontStyle="bold" fill={C.ego} listening={false} />}
          {p.selection?.kind === 'ego' && handle(ego.x, ego.y, ego.yaw, L, 'ego')}
        </Layer>
      </Stage>
      <div className="map-hud">
        <span>{cursor ? `x ${cursor[0].toFixed(1)} m · y ${cursor[1].toFixed(1)} m` : 'road frame · x forward · y left'}</span>
        <ScaleBar s={view.s} />
      </div>
    </div>
  )
}

function ScaleBar({ s }: { s: number }) {
  const meters = [1, 2, 5, 10, 20, 50, 100].find(m => m * s >= 60) ?? 100
  return <span className="scale"><i style={{ width: meters * s }} />{meters} m</span>
}
