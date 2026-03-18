import { useRef } from 'react';
import type { VODItem } from '../store/useAppStore';
import { VODCard } from './VODCard';
import './VODRow.css';

interface VODRowProps {
  title: string;
  items: VODItem[];
  showRank?: boolean;
  highlight?: boolean;
  layout?: 'scroll' | 'grid';
}

export function VODRow({ title, items, showRank, highlight, layout = 'scroll' }: VODRowProps) {
  const rowRef = useRef<HTMLDivElement>(null);

  const scroll = (dir: 'left' | 'right') => {
    rowRef.current?.scrollBy({ left: dir === 'left' ? -480 : 480, behavior: 'smooth' });
  };

  if (items.length === 0) return null;

  return (
    <section className={`vod-row ${highlight ? 'vod-row--highlight' : ''}`}>
      <div className="vod-row-header">
        <h2 className="vod-row-title">
          {highlight && <span className="row-highlight-dot" />}
          {title}
        </h2>
        {highlight && <span className="badge-recommend">AI 맞춤 추천</span>}
      </div>

      {layout === 'grid' ? (
        <div className="vod-row-grid">
          {items.map((vod, idx) => (
            <VODCard key={vod.id} vod={vod} rank={showRank ? idx + 1 : undefined} />
          ))}
        </div>
      ) : (
        <div className="vod-row-scroll-wrapper">
          <button
            type="button"
            className="row-scroll-btn row-scroll-btn--left"
            onClick={() => scroll('left')}
          >
            ‹
          </button>
          <div className="vod-row-items" ref={rowRef}>
            {items.map((vod, idx) => (
              <VODCard key={vod.id} vod={vod} rank={showRank ? idx + 1 : undefined} />
            ))}
          </div>
          <button
            type="button"
            className="row-scroll-btn row-scroll-btn--right"
            onClick={() => scroll('right')}
          >
            ›
          </button>
        </div>
      )}
    </section>
  );
}
