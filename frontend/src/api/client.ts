const API_BASE = '/api';

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { error?: string }).error || `API error ${res.status}`);
  }
  return res.json();
}

export const api = {
  chat: (body: { message: string; history?: { role: string; content: string }[]; context?: object }) =>
    fetchApi<{ reply: string }>('/chat', { method: 'POST', body: JSON.stringify(body) }),

  emotion: (body: { text: string }) =>
    fetchApi<{ emotion: string; confidence: number }>('/emotion', { method: 'POST', body: JSON.stringify(body) }),

  recommend: (body: { emotion?: string; watchHistory?: string[]; lastVodId?: string; limit?: number }) =>
    fetchApi<{ items: { id: string; title: string; genre: string; description?: string; popularity?: number; featured?: boolean; rating?: string; year?: number }[]; message?: string }>('/recommend', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  vodCatalog: () =>
    fetchApi<{ items: { id: string; title: string; genre: string; description?: string; popularity?: number; featured?: boolean; rating?: string; year?: number }[] }>('/vod/catalog'),

  vodAnalysis: (body: { vodId: string; vodTitle: string; genre?: string }) =>
    fetchApi<{ question: string; keywords: string[] }>('/vod/analysis', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
};
