import type { ReactNode } from 'react'

export function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="section">
      <header><h3>{title}</h3>{aside}</header>
      <div className="section-body">{children}</div>
    </section>
  )
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}{hint && <em>{hint}</em>}</span>
      {children}
    </label>
  )
}

export function Num({ value, onChange, step = 1, min, max, unit }: {
  value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number; unit?: string
}) {
  return (
    <div className="num">
      <input type="number" value={Number.isFinite(value) ? value : ''} step={step} min={min} max={max}
             onChange={e => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) onChange(v) }} />
      {unit && <span>{unit}</span>}
    </div>
  )
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button type="button" className={`toggle ${checked ? 'on' : ''}`} onClick={() => onChange(!checked)}>
      <i />{label}
    </button>
  )
}

export function Segmented<T extends string>({ value, options, onChange }: {
  value: T; options: { id: T; label: string }[]; onChange: (v: T) => void
}) {
  return (
    <div className="segmented">
      {options.map(o => <button type="button" key={o.id} className={o.id === value ? 'active' : ''} onClick={() => onChange(o.id)}>{o.label}</button>)}
    </div>
  )
}

export const deg = (rad: number) => Math.round(rad * 1800 / Math.PI) / 10
export const rad = (d: number) => d * Math.PI / 180
