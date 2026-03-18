import { useState } from 'react';
import type { VODItem } from '../store/useAppStore';
import { useServiceLoop } from '../hooks/useServiceLoop';
import { VODPlayer } from './VODPlayer';
import './HeroBanner.css';

interface HeroBannerProps {
  vod: VODItem;
  onConversation: () => void;
}

export function HeroBanner({ vod, onConversation }: HeroBannerProps) {
  const [playing, setPlaying] = useState(false);
  const { onVodEnd } = useServiceLoop();

  if (playing) {
    return (
      <VODPlayer
        vod={vod}
        onEnd={() => { setPlaying(false); onVodEnd(vod.id, vod.title, vod.genre); }}
        onBack={() => setPlaying(false)}
      />
    );
  }

  return (
    <div className="hero-banner">
      <div className="hero-bg" style={{ '--genre': getGenreVar(vod.genre) } as React.CSSProperties} />
      <div className="hero-overlay" />
      <div className="hero-content">
        <div className="hero-badges">
          <span className="hero-badge badge-featured">에디터 픽</span>
          <span className="hero-badge badge-genre">{vod.genre}</span>
          {vod.rating && <span className="hero-badge badge-rating">★ {vod.rating}</span>}
          {vod.year && <span className="hero-badge badge-year">{vod.year}</span>}
        </div>
        <h1 className="hero-title">{vod.title}</h1>
        <p className="hero-desc">{vod.description}</p>
        <div className="hero-actions">
          <button type="button" className="btn-hero-play" onClick={() => setPlaying(true)}>
            ▶ 지금 시청하기
          </button>
          <button type="button" className="btn-hero-ai" onClick={onConversation}>
            ♥ 마음이음 추천 받기
          </button>
        </div>
      </div>
    </div>
  );
}

function getGenreVar(genre: string): string {
  const map: Record<string, string> = {
    '드라마': '#0f2657, #1d3a8a',
    '영화':   '#1a0a40, #4c1d95',
    '예능':   '#2a0a4c, #6b21a8',
    '로맨스': '#3d0a1e, #831843',
    '멜로':   '#3b0f24, #9d174d',
    '액션':   '#3b0a0a, #991b1b',
    '스릴러': '#2a1200, #78350f',
    '다큐':   '#062c2c, #134e4a',
  };
  return map[genre] ?? '#0f1629, #1e2a4a';
}
