import './Sidebar.css';

export type NavView = 'home' | 'recommend' | 'popular';

const GENRES = ['전체', '드라마', '영화', '예능', '로맨스', '멜로', '액션', '스릴러', '다큐'];

interface SidebarProps {
  navView: NavView;
  onNavChange: (view: NavView) => void;
  selectedGenre: string;
  onGenreChange: (genre: string) => void;
  onConversation: () => void;
}

export function Sidebar({ navView, onNavChange, selectedGenre, onGenreChange, onConversation }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="logo-icon">♥</span>
        <span className="logo-text">마음이음</span>
      </div>

      <nav className="sidebar-nav">
        <button
          type="button"
          className={`nav-item ${navView === 'home' && selectedGenre === '전체' ? 'active' : ''}`}
          onClick={() => { onNavChange('home'); onGenreChange('전체'); }}
        >
          <span className="nav-icon">⊞</span>
          <span>홈</span>
        </button>
        <button
          type="button"
          className={`nav-item ${navView === 'recommend' ? 'active' : ''}`}
          onClick={() => onNavChange('recommend')}
        >
          <span className="nav-icon">✦</span>
          <span>내 추천</span>
        </button>
        <button
          type="button"
          className={`nav-item ${navView === 'popular' ? 'active' : ''}`}
          onClick={() => onNavChange('popular')}
        >
          <span className="nav-icon">↑</span>
          <span>인기 TOP</span>
        </button>
      </nav>

      <div className="sidebar-divider" />

      <div className="sidebar-genre-label">장르</div>
      <nav className="sidebar-genres">
        {GENRES.map((genre) => (
          <button
            key={genre}
            type="button"
            className={`genre-item ${selectedGenre === genre && navView === 'home' ? 'active' : ''}`}
            onClick={() => onGenreChange(genre)}
          >
            <span className={`genre-dot genre-dot--${genre}`} />
            {genre}
          </button>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <button type="button" className="btn-ai-call" onClick={onConversation}>
          <span className="ai-icon">♥</span>
          마음이음 호출
        </button>
      </div>
    </aside>
  );
}
