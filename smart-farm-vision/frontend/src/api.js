const API_BASE =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? 'http://127.0.0.1:8000' : window.location.origin);

export function mediaUrl(path) {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  return `${API_BASE}${path}`;
}

async function request(path, options = {}) {
  const token = localStorage.getItem('agrovision_token');
  const headers = new Headers(options.headers || {});
  if (token) headers.set('Authorization', `Bearer ${token}`);
  if (options.json) {
    headers.set('Content-Type', 'application/json');
    options.body = JSON.stringify(options.json);
  }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.detail || 'Ошибка запроса');
    error.status = response.status;
    throw error;
  }
  return data;
}

export const api = {
  login: (payload) => request('/api/auth/login', { method: 'POST', json: payload }),
  me: () => request('/api/me'),
  profiles: () => request('/api/profiles'),
  history: () => request('/api/history'),
  stats: () => request('/api/stats'),
  exportHistory: async () => {
    const token = localStorage.getItem('agrovision_token');
    const response = await fetch(`${API_BASE}/api/history/export`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!response.ok) throw new Error('Не удалось выгрузить историю');
    return response.blob();
  },
  exportReport: async (analysisId) => {
    const token = localStorage.getItem('agrovision_token');
    const response = await fetch(`${API_BASE}/api/analysis/${analysisId}/report`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!response.ok) throw new Error('Не удалось сформировать отчет');
    return response.blob();
  },
  pay: () => request('/api/payment/activate', { method: 'POST' }),
  analyze: ({ profileId, confidence, file }) => {
    const body = new FormData();
    body.append('profile_id', profileId);
    body.append('confidence', String(confidence));
    body.append('file', file);
    return request('/api/analyze', { method: 'POST', body });
  },
  analyzeBatch: ({ profileId, confidence, files }) => {
    const body = new FormData();
    body.append('profile_id', profileId);
    body.append('confidence', String(confidence));
    for (const file of files) body.append('files', file);
    return request('/api/analyze-batch', { method: 'POST', body });
  },
};
