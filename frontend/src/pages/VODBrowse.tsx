import { useEffect, useState, useMemo } from 'react';
import { api } from '../api/client';
import { useAppStore } from '../store/useAppStore';
import { Sidebar } from '../components/Sidebar';
import type { NavView } from '../components/Sidebar';
import { SearchBar } from '../components/SearchBar';
import { HeroBanner } from '../components/HeroBanner';
import { VODRow } from '../components/VODRow';
import { VODCard } from '../components/VODCard';
import { ConversationPanel } from '../components/ConversationPanel';
import { useKeyboardNav } from '../hooks/useKeyboardNav';
import './VODBrowse.css';

export function VODBrowse() {
  const {
    state,
    setState,
    vodCatalog,
    setVodCatalog,
    recommendedItems,
  } = useAppStore();

  const [loading, setLoading] = useState(true);
  const [navView, setNavView] = useState<NavView>('home');
  const [selectedGenre, setSelectedGenre] = useState('전체');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    api.vodCatalog().then(({ items }) => {
      setVodCatalog(items);
      setLoading(false);
    });
  }, [setVodCatalog]);

  useKeyboardNav({
    onBack: () => state === 'conversation' && setState('vod_browse'),
  });

  const popularItems = useMemo(
    () => [...vodCatalog].sort((a, b) => (b.popularity ?? 0) - (a.popularity ?? 0)),
    [vodCatalog]
  );

  const featuredVod = useMemo(
    () => vodCatalog.find((v) => v.featured) ?? vodCatalog[0],
    [vodCatalog]
  );

  const searchResults = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return [];
    return vodCatalog.filter(
      (v) =>
        v.title.toLowerCase().includes(q) ||
        v.genre.toLowerCase().includes(q) ||
        (v.description ?? '').toLowerCase().includes(q)
    );
  }, [searchQuery, vodCatalog]);

  const filteredByGenre = useMemo(() => {
    if (selectedGenre === '전체') return vodCatalog;
    return vodCatalog.filter((v) => v.genre === selectedGenre);
  }, [vodCatalog, selectedGenre]);

  const genreRows = useMemo(() => {
    const genres = ['드라마', '영화', '예능', '로맨스', '액션', '멜로', '스릴러', '다큐'];
    return genres
      .map((genre) => ({ genre, items: vodCatalog.filter((v) => v.genre === genre) }))
      .filter((g) => g.items.length > 0);
  }, [vodCatalog]);

  const isDimmed = state === 'conversation';

  const handleNavChange = (view: NavView) => {
    setNavView(view);
    setSelectedGenre('전체');
    setSearchQuery('');
    if (state === 'recommendation') setState('vod_browse');
  };

  const handleGenreChange = (genre: string) => {
    setSelectedGenre(genre);
    setNavView('home');
    setSearchQuery('');
    if (state === 'recommendation') setState('vod_browse');
  };

  const renderMain = () => {
    if (loading) {
      return (
        <div className="loading-screen">
          <div className="loading-spinner" />
          <p>콘텐츠를 불러오는 중...</p>
        </div>
      );
    }

    if (searchQuery.trim()) {
      return (
        <div className="search-result-section">
          <div className="search-result-header">
            <span className="search-keyword">"{searchQuery}"</span> 검색 결과
            <span className="search-count">{searchResults.length}건</span>
          </div>
          {searchResults.length === 0 ? (
            <p className="empty-msg">검색 결과가 없습니다.</p>
          ) : (
            <div className="search-result-grid">
              {searchResults.map((vod) => (
                <VODCard key={vod.id} vod={vod} />
              ))}
            </div>
          )}
        </div>
      );
    }

    if (state === 'recommendation' && recommendedItems.length > 0) {
      return (
        <>
          <VODRow title="내 맞춤 추천" items={recommendedItems} highlight />
          <VODRow title="인기 TOP 10" items={popularItems.slice(0, 10)} showRank />
        </>
      );
    }

    if (navView === 'popular') {
      return (
        <VODRow title="인기 TOP 25" items={popularItems} showRank layout="grid" />
      );
    }

    if (navView === 'recommend') {
      return (
        <>
          {recommendedItems.length > 0 ? (
            <VODRow title="내 맞춤 추천" items={recommendedItems} highlight />
          ) : (
            <div className="recommend-empty">
              <p className="recommend-empty-title">아직 추천이 없어요</p>
              <p className="recommend-empty-desc">마음이음과 대화하면 맞춤 추천을 받을 수 있어요!</p>
              <button
                type="button"
                className="btn-start-chat"
                onClick={() => setState('conversation')}
              >
                ♥ 마음이음 대화 시작
              </button>
            </div>
          )}
          <VODRow title="지금 인기 있는 콘텐츠" items={popularItems.slice(0, 10)} showRank />
        </>
      );
    }

    if (selectedGenre !== '전체') {
      return (
        <VODRow
          title={`${selectedGenre} 콘텐츠`}
          items={filteredByGenre}
          layout="grid"
        />
      );
    }

    return (
      <>
        {featuredVod && (
          <HeroBanner
            vod={featuredVod}
            onConversation={() => setState('conversation')}
          />
        )}
        <VODRow title="지금 인기 있는 콘텐츠" items={popularItems.slice(0, 10)} showRank />
        {recommendedItems.length > 0 && (
          <VODRow title="내 맞춤 추천" items={recommendedItems} highlight />
        )}
        {genreRows.map(({ genre, items }) => (
          <VODRow key={genre} title={genre} items={items} />
        ))}
      </>
    );
  };

  return (
    <div className="app-layout">
      <Sidebar
        navView={navView}
        onNavChange={handleNavChange}
        selectedGenre={selectedGenre}
        onGenreChange={handleGenreChange}
        onConversation={() => setState('conversation')}
      />

      <div className="app-content">
        <header className="app-header">
          <SearchBar value={searchQuery} onChange={setSearchQuery} />
        </header>

        <main className={`app-main ${isDimmed ? 'app-main--dimmed' : ''}`}>
          {renderMain()}
        </main>
      </div>

      {state === 'conversation' && (
        <ConversationPanel
          onClose={() => setState('vod_browse')}
          onShowRecommend={() => setState('recommendation')}
        />
      )}
    </div>
  );
}
