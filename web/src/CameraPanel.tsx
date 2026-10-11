import { Camera, RefreshCw } from 'lucide-react'
import type { Preview } from './types'

type Props = { preview: Preview; sceneId: string; onGenerate: () => void }

export default function CameraPanel({ preview, sceneId, onGenerate }: Props) {
  if (preview.status !== 'ready' || !preview.base_url || !preview.cameras) {
    return (
      <div className="empty-preview">
        <Camera size={34} />
        <h3>Six-camera preview</h3>
        <p>{preview.status === 'running' ? 'Rendering preview…' : 'Generate a low-resolution view of this trained scene.'}</p>
        <button className="primary" onClick={onGenerate} disabled={preview.status === 'running'}>
          <RefreshCw size={15} className={preview.status === 'running' ? 'spin' : ''} />
          {preview.status === 'running' ? 'Rendering' : 'Generate preview'}
        </button>
      </div>
    )
  }
  const order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
  return (
    <div className="camera-grid">
      {order.map(name => preview.cameras?.[name] && (
        <figure key={name}>
          <img src={`${preview.base_url}/${preview.cameras[name]}?scene=${sceneId}`} alt={name} />
          <figcaption>{name.replace('CAM_', '').replace('_', ' ')}</figcaption>
        </figure>
      ))}
    </div>
  )
}
