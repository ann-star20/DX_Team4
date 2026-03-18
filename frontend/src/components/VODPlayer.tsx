import type { VODItem } from '../store/useAppStore';
import './VODPlayer.css';

interface VODPlayerProps {
  vod: VODItem;
  onEnd: () => void;
  onBack: () => void;
}

export function VODPlayer({ vod, onEnd, onBack }: VODPlayerProps) {
  return (
    <div className="vod-player-overlay">
      <div className="vod-player">
        <div className="vod-player-screen">
          <p>시청 중: {vod.title}</p>
          <p className="hint">(프로토타입: 시뮬레이션)</p>
          <button type="button" className="btn-end" onClick={onEnd}>
            시청 종료
          </button>
        </div>
        <button type="button" className="btn-back" onClick={onBack}>
          ← 목록으로
        </button>
      </div>
    </div>
  );
}
