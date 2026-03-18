import { useState } from 'react';
import type { VODItem } from '../store/useAppStore';
import { useServiceLoop } from '../hooks/useServiceLoop';
import { VODPlayer } from './VODPlayer';
import './VODCard.css';

const GENRE_COLORS: Record<string, [string, string]> = {
  '드라마':  ['#0c1f52', '#1d4ed8'],
  '영화':    ['#1a0840', '#6d28d9'],
  '예능':    ['#26084c', '#7c3aed'],
  '로맨스':  ['#3d091e', '#be185d'],
  '멜로':    ['#3b0f24', '#9d174d'],
  '액션':    ['#3b0a0a', '#b91c1c'],
  '스릴러':  ['#2a1200', '#92400e'],
  '다큐':    ['#062c2c', '#0f766e'],
};

function getGenreGradient(genre: string): string {
  const [from, to] = GENRE_COLORS[genre] ?? ['#111224', '#1e2240'];
  return `linear-gradient(135deg, ${from} 0%, ${to} 100%)`;
}

interface VODCardProps {
  vod: VODItem;
  rank?: number;
}

export function VODCard({ vod, rank }: VODCardProps) {
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
    <button type="button" className="vod-card" onClick={() => setPlaying(true)} tabIndex={0}>
      <div className="vod-card-thumb" style={{ background: getGenreGradient(vod.genre) }}>
        {rank !== undefined && rank <= 10 && (
          <span className={`rank-badge ${rank <= 3 ? 'rank-badge--top' : ''}`}>
            #{rank}
          </span>
        )}
        <div className="vod-card-hover-overlay">
          <span className="play-btn-icon">▶</span>
        </div>
        {vod.rating && (
          <span className="rating-badge">★ {vod.rating}</span>
        )}
      </div>
      <div className="vod-card-info">
        <div className="vod-card-meta">
          <span className={`genre-badge genre-badge--${vod.genre}`}>{vod.genre}</span>
          {vod.year && <span className="year-text">{vod.year}</span>}
        </div>
        <h3 className="vod-card-title">{vod.title}</h3>
        {vod.description && (
          <p className="vod-card-desc">{vod.description}</p>
        )}
      </div>
    </button>
  );
}
