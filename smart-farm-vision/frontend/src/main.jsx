import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  BadgeCheck,
  Camera,
  ChevronRight,
  CreditCard,
  Download,
  Database,
  Eye,
  FileVideo,
  FileText,
  Leaf,
  LineChart,
  Loader2,
  LockKeyhole,
  LogOut,
  Search,
  Sprout,
  UploadCloud,
} from 'lucide-react';
import { api, mediaUrl } from './api.js';
import './styles.css';

const GROUP_ORDER = ['cucumber', 'tomato', 'lettuce', 'cucumber_leaf', 'strawberry'];

const GROUP_COPY = {
  cucumber: {
    icon: Sprout,
    title: 'Огурцы',
    text: 'Плоды, завязи и сегментация прямые/искривленные плоды',
  },
  tomato: {
    icon: Camera,
    title: 'Томаты',
    text: 'Детекция, оценка степени зрелости и подсчет плодов',
  },
  lettuce: {
    icon: Leaf,
    title: 'Листовой салат',
    text: 'Здоровые листья и признаки дефицита питания',
  },
  cucumber_leaf: {
    icon: LineChart,
    title: 'Листья огурца',
    text: 'Детекция и подсчет количества листьев',
  },
  strawberry: {
    icon: BadgeCheck,
    title: 'Клубника',
    text: 'Детекция ягод и подсчет количества',
  },
};

function App() {
  const [token, setToken] = useState(localStorage.getItem('agrovision_token'));
  const [user, setUser] = useState(null);
  const [profiles, setProfiles] = useState([]);
  const [activeGroup, setActiveGroup] = useState('cucumber');
  const [activeProfileId, setActiveProfileId] = useState('cucumber');
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);
  const [files, setFiles] = useState([]);
  const [confidence, setConfidence] = useState(0.25);
  const [result, setResult] = useState(null);
  const [batchItems, setBatchItems] = useState([]);
  const [batchMeta, setBatchMeta] = useState(null);
  const [selectedBatchResultId, setSelectedBatchResultId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [paymentOpen, setPaymentOpen] = useState(false);

  useEffect(() => {
    api.profiles().then(setProfiles).catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!token) return;
    refreshUser();
    refreshHistory();
    refreshStats();
  }, [token]);

  const grouped = useMemo(() => {
    const map = {};
    for (const profile of profiles) {
      const key = profile.group || profile.id;
      map[key] = map[key] || [];
      map[key].push(profile);
    }
    return map;
  }, [profiles]);

  const activeProfiles = grouped[activeGroup] || [];
  const activeProfile = profiles.find((item) => item.id === activeProfileId) || activeProfiles[0];
  const confidenceMin = activeProfile?.confidence_min ?? 0.05;
  const confidenceMax = activeProfile?.confidence_max ?? 0.95;
  const confidenceStep = activeProfile?.confidence_step ?? 0.05;
  const confidenceEnabled = activeProfile?.mode === 'YOLO';

  useEffect(() => {
    const first = grouped[activeGroup]?.[0];
    if (first && !grouped[activeGroup].some((item) => item.id === activeProfileId)) {
      setActiveProfileId(first.id);
    }
  }, [activeGroup, grouped, activeProfileId]);

  useEffect(() => {
    if (!activeProfile) return;
    setConfidence(activeProfile.confidence_default ?? 0.25);
  }, [activeProfile?.id]);

  async function refreshUser() {
    try {
      setUser(await api.me());
    } catch {
      logout();
    }
  }

  async function refreshHistory() {
    setHistory(await api.history());
  }

  async function refreshStats() {
    setStats(await api.stats());
  }

  function logout() {
    localStorage.removeItem('agrovision_token');
    setToken(null);
    setUser(null);
    setHistory([]);
    setStats(null);
  }

  async function handleAnalyze(event) {
    event.preventDefault();
    if (!files.length || !activeProfile) return;
    setBusy(true);
    setError('');
    try {
      const data = files.length === 1
        ? await api.analyze({ profileId: activeProfile.id, confidence, file: files[0] })
        : await api.analyzeBatch({ profileId: activeProfile.id, confidence, files });
      if (data.items) {
        const successfulItems = data.items.filter((item) => !item.error);
        const failedItems = data.items.filter((item) => item.error);
        const primaryResult = successfulItems[0] || null;
        setBatchItems(successfulItems);
        setSelectedBatchResultId(primaryResult?.id ?? null);
        setBatchMeta({
          total: data.items.length,
          successCount: successfulItems.length,
          failedItems,
        });
        if (failedItems.length) {
          setError(`Успешно обработано ${successfulItems.length} из ${data.items.length} файлов. Остальные не прошли обработку.`);
        }
        setResult(primaryResult);
      } else {
        setBatchItems([]);
        setBatchMeta(null);
        setSelectedBatchResultId(null);
        setResult(data);
      }
      setFiles([]);
      await refreshUser();
      await refreshHistory();
      await refreshStats();
    } catch (err) {
      if (err.status === 402) setPaymentOpen(true);
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handlePayment() {
    const paid = await api.pay();
    setUser(paid);
    setPaymentOpen(false);
    setError('');
  }

  async function handleExportHistory() {
    const blob = await api.exportHistory();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'agrovision_history.csv';
    link.click();
    URL.revokeObjectURL(url);
  }

  async function handleDownloadReport(item) {
    if (!item?.id) return;
    const blob = await api.exportReport(item.id);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `agrovision_report_${item.id}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function handleOpenHistory(row) {
    setBatchItems([]);
    setBatchMeta(null);
    setSelectedBatchResultId(null);
    const profile = profiles.find((item) => item.id === row.profile_id);
    if (profile?.group) setActiveGroup(profile.group);
    if (profile?.id) setActiveProfileId(profile.id);
    setResult({
      id: row.id,
      profile_id: row.profile_id,
      media_type: row.media_type,
      original_name: row.original_name,
      counts: safeCounts(row.counts ?? row.counts_json),
      summary: row.summary,
      recommendation: row.recommendation,
      elapsed_ms: row.elapsed_ms,
      output_url: row.output_url,
      report_url: row.report_url,
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  if (!token) {
    return <LoginScreen setToken={setToken} setUser={setUser} />;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Leaf size={22} /></div>
          <div>
            <strong>AgroVision Lab</strong>
            <span>умная городская ферма</span>
          </div>
        </div>

        <nav className="culture-nav">
          {GROUP_ORDER.map((group) => {
            const Icon = GROUP_COPY[group].icon;
            const available = grouped[group]?.some((profile) => profile.model_available);
            return (
              <button
                key={group}
                className={group === activeGroup ? 'nav-item active' : 'nav-item'}
                onClick={() => setActiveGroup(group)}
              >
                <Icon size={18} />
                <span>{GROUP_COPY[group].title}</span>
                {available && <i />}
              </button>
            );
          })}
        </nav>

        <div className="quota">
          <span>Доступ</span>
          <strong>{user?.is_paid ? 'Подписка активна' : `${user?.remaining_uploads ?? 0} из 3 загрузок`}</strong>
          {!user?.is_paid && (
            <i className="quota-meter">
              <b style={{ '--used': `${Math.min(100, ((user?.used_uploads ?? 0) / (user?.free_upload_limit || 3)) * 100)}%` }} />
            </i>
          )}
          {!user?.is_paid && (
            <button className="pay-link" onClick={() => setPaymentOpen(true)}>
              <CreditCard size={16} /> Оплатить
            </button>
          )}
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>Комплекс моделей компьютерного зрения</h1>
            <p>{user?.company}</p>
          </div>
          <button className="ghost-button" onClick={logout}>
            <LogOut size={17} /> Выйти
          </button>
        </header>

        <section className="slogan-row">
          <Metric icon={Camera} title="Быстрая оценка" text="состояния плодовых растений" />
          <Metric icon={LineChart} title="Эффективность" text="сотрудников фермы" />
          <Metric icon={Database} title="История" text="результатов в базе данных" />
        </section>

        <StatsPanel stats={stats} />

        <section className="work-grid">
          <div className="panel analyze-panel">
            <div className="panel-heading">
              <div>
                <h2>{GROUP_COPY[activeGroup].title}</h2>
                <p>{GROUP_COPY[activeGroup].text}</p>
              </div>
              <span className={activeProfile?.model_available ? 'status ready' : 'status'}>
                {activeProfile?.model_available ? 'YOLO подключена' : 'Demo CV'}
              </span>
            </div>

            {activeProfiles.length > 1 && (
              <div className="segmented">
                {activeProfiles.map((profile) => (
                  <button
                    key={profile.id}
                    className={profile.id === activeProfileId ? 'selected' : ''}
                    onClick={() => setActiveProfileId(profile.id)}
                  >
                    {profile.short_title}
                  </button>
                ))}
              </div>
            )}

            {activeProfile?.mode_note && (
              <div className="context-note">
                <strong>Режим профиля</strong>
                <p>{activeProfile.mode_note}</p>
              </div>
            )}

            <div className="model-facts">
              <div>
                <span>Режим</span>
                <strong>{activeProfile?.mode === 'YOLO' ? 'YOLO inference' : 'Demo CV'}</strong>
              </div>
              <div>
                <span>Материал</span>
                <strong>Фото / видео</strong>
              </div>
              <div>
                <span>Порог</span>
                <strong>{confidenceEnabled ? `${Math.round(confidence * 100)}%` : 'универсальный'}</strong>
              </div>
            </div>

            <form onSubmit={handleAnalyze}>
              <label className="dropzone">
                <input
                  type="file"
                  accept="image/*,video/*"
                  multiple
                  onChange={(event) => setFiles(Array.from(event.target.files || []))}
                />
                <UploadCloud size={30} />
                <strong>{files.length ? `${files.length} файл(ов) выбрано` : 'Загрузить фото или видео'}</strong>
                <span>{activeProfile?.task}</span>
              </label>

              {files.length > 0 && (
                <div className="file-list">
                  {files.slice(0, 5).map((item) => <span key={item.name}>{item.name}</span>)}
                  {files.length > 5 && <span>Еще файлов: {files.length - 5}</span>}
                </div>
              )}

              {confidenceEnabled ? (
                <>
                  <div className="control-row">
                    <label>
                      Порог уверенности
                      <input
                        type="range"
                        min={confidenceMin}
                        max={confidenceMax}
                        step={confidenceStep}
                        value={confidence}
                        onChange={(event) => setConfidence(Number(event.target.value))}
                      />
                    </label>
                    <b>{Math.round(confidence * 100)}%</b>
                  </div>
                  <div className="control-help">
                    {activeProfile?.confidence_note && <p>{activeProfile.confidence_note}</p>}
                    {activeProfile?.confidence_range_note && <p>{activeProfile.confidence_range_note}</p>}
                  </div>
                </>
              ) : (
                <div className="control-help demo-help">
                  <p>{activeProfile?.confidence_note || 'Применено универсальное значение порога'}</p>
                  {activeProfile?.confidence_range_note && <p>{activeProfile.confidence_range_note}</p>}
                </div>
              )}

              {error && <div className="error">{error}</div>}

              <button className="primary-button" disabled={!files.length || busy}>
                {busy ? <Loader2 className="spin" size={18} /> : <ChevronRight size={18} />}
                {files.length > 1 ? 'Запустить пакетный анализ' : 'Запустить анализ'}
              </button>
            </form>
          </div>

          <ResultPanel
            result={result}
            activeProfile={activeProfile}
            batchItems={batchItems}
            batchMeta={batchMeta}
            selectedBatchResultId={selectedBatchResultId}
            onSelectBatch={(item) => {
              setSelectedBatchResultId(item.id);
              setResult(item);
            }}
            onReport={handleDownloadReport}
          />
        </section>

        <HistoryPanel
          history={history}
          profiles={profiles}
          onExport={handleExportHistory}
          onOpen={handleOpenHistory}
          onReport={handleDownloadReport}
        />
      </main>

      {paymentOpen && (
        <PaymentModal onClose={() => setPaymentOpen(false)} onPayment={handlePayment} company={user?.company} />
      )}
    </div>
  );
}

function LoginScreen({ setToken, setUser }) {
  const [company, setCompany] = useState('ТюмГУ Smart Farm');
  const [password, setPassword] = useState('demo');
  const [error, setError] = useState('');

  async function submit(event) {
    event.preventDefault();
    setError('');
    try {
      const data = await api.login({ company, password });
      localStorage.setItem('agrovision_token', data.token);
      setToken(data.token);
      setUser(data);
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <main className="login-screen">
      <section className="login-visual">
        <div className="brand large">
          <div className="brand-mark"><Leaf size={28} /></div>
          <div>
            <strong>AgroVision Lab</strong>
            <span>программная платформа</span>
          </div>
        </div>
        <h1>Комплекс моделей компьютерного зрения для умной городской фермы</h1>
        <div className="login-points">
          <span>Быстрая оценка состояния плодовых растений</span>
          <span>Повышайте эффективность сотрудников фермы</span>
          <span>Разработаем решение под вашу теплицу</span>
        </div>
      </section>

      <form className="login-card" onSubmit={submit}>
        <LockKeyhole size={24} />
        <h2>Вход в кабинет</h2>
        <label>
          Название компании
          <input value={company} onChange={(event) => setCompany(event.target.value)} />
        </label>
        <label>
          Пароль
          <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
        </label>
        {error && <div className="error">{error}</div>}
        <button className="primary-button">Войти</button>
      </form>
    </main>
  );
}

function Metric({ icon: Icon, title, text }) {
  return (
    <div className="metric">
      <Icon size={19} />
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}

function StatsPanel({ stats }) {
  const profileEntries = Object.entries(stats?.by_profile || {});
  const objectEntries = Object.entries(stats?.objects || {}).slice(0, 6);
  const maxProfile = Math.max(1, ...profileEntries.map(([, value]) => value));
  const maxObject = Math.max(1, ...objectEntries.map(([, value]) => value));

  return (
    <section className="panel stats-panel">
      <div className="panel-heading compact">
        <div>
          <h2>Сводные данные мониторинга</h2>
          <p>Графики формируются по сохраненной истории обработок в базе данных</p>
        </div>
        <LineChart size={21} />
      </div>
      <div className="stats-grid">
        <div className="stat-card">
          <span>Обработок</span>
          <strong>{stats?.total_analyses ?? 0}</strong>
        </div>
        <div className="stat-card">
          <span>Общее время</span>
          <strong>{stats?.avg_elapsed_ms ?? 0} мс</strong>
        </div>
        <div className="bar-box">
          <b>По направлениям</b>
          {profileEntries.length ? profileEntries.map(([label, value]) => (
            <Bar key={label} label={label} value={value} max={maxProfile} />
          )) : <span className="muted-inline">Нет данных</span>}
        </div>
        <div className="bar-box">
          <b>Найденные объекты</b>
          {objectEntries.length ? objectEntries.map(([label, value]) => (
            <Bar key={label} label={label} value={value} max={maxObject} />
          )) : <span className="muted-inline">Нет данных</span>}
        </div>
      </div>
    </section>
  );
}

function Bar({ label, value, max }) {
  return (
    <div className="bar-row">
      <span>{label}</span>
      <i style={{ '--w': `${Math.max(7, (value / max) * 100)}%` }} />
      <strong>{value}</strong>
    </div>
  );
}

function ResultPanel({ result, activeProfile, batchItems, batchMeta, selectedBatchResultId, onSelectBatch, onReport }) {
  const countEntries = Object.entries(result?.counts || {});

  return (
    <div className="panel result-panel">
      <div className="panel-heading">
        <div>
          <h2>Результат</h2>
          <p>{activeProfile?.title || 'Выберите модель'}</p>
        </div>
        <FileVideo size={22} />
      </div>

      {!result && (
        <div className="empty-result">
          <Camera size={44} />
          <strong>После анализа здесь появятся разметка и отчет.</strong>
        </div>
      )}

      {result && (
        <>
          {batchMeta && (
            <div className="batch-panel">
              <div className="batch-panel__head">
                <strong>Пакетный анализ</strong>
                <span>Успешно обработано {batchMeta.successCount} из {batchMeta.total}</span>
              </div>
              {batchItems.length > 0 && (
                <div className="batch-switcher">
                  {batchItems.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      className={item.id === selectedBatchResultId ? 'batch-chip is-active' : 'batch-chip'}
                      onClick={() => onSelectBatch(item)}
                    >
                      {item.original_name}
                    </button>
                  ))}
                </div>
              )}
              {batchMeta.failedItems.length > 0 && (
                <div className="batch-errors">
                  {batchMeta.failedItems.map((item, index) => (
                    <span key={`${item.original_name}-${index}`}>
                      Не обработан: {item.original_name}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
          {!batchMeta && batchItems.length > 1 && (
            <div className="file-list">
              {batchItems.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={item.id === selectedBatchResultId ? 'ghost-button compact-button' : 'ghost-button compact-button'}
                  onClick={() => onSelectBatch(item)}
                >
                  {item.original_name}
                </button>
              ))}
            </div>
          )}
          <div className="result-meta">
            <span>{result.original_name || activeProfile?.title}</span>
            <span>{result.elapsed_ms ? `${result.elapsed_ms} мс` : 'время не указано'}</span>
            <button className="ghost-button compact-button" onClick={() => onReport(result)}>
              <FileText size={16} /> TXT-отчет
            </button>
          </div>
          <div className="preview">
            {result.media_type === 'video' ? (
              <video src={mediaUrl(result.output_url)} controls />
            ) : (
              <img src={mediaUrl(result.output_url)} alt="Результат анализа" />
            )}
          </div>
          <div className="count-list">
            {countEntries.length ? countEntries.map(([key, value]) => (
              <div key={key}>
                <span>{key}</span>
                <strong>{value}</strong>
              </div>
            )) : (
              <div>
                <span>Объекты</span>
                <strong>0</strong>
              </div>
            )}
          </div>
          <p className="summary">{result.summary}</p>
          {result.recommendation && (
            <div className="recommendation">
              <strong>Рекомендация</strong>
              <p>{result.recommendation}</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function HistoryPanel({ history, profiles, onExport, onOpen, onReport }) {
  const [query, setQuery] = useState('');
  const [profileFilter, setProfileFilter] = useState('all');
  const profileLabels = Object.fromEntries(profiles.map((profile) => [profile.id, profile.short_title || profile.title]));
  const filteredHistory = history.filter((row) => {
    const haystack = `${row.original_name} ${row.profile_id} ${formatCounts(row.counts ?? row.counts_json)}`.toLowerCase();
    const matchesQuery = haystack.includes(query.trim().toLowerCase());
    const matchesProfile = profileFilter === 'all' || row.profile_id === profileFilter;
    return matchesQuery && matchesProfile;
  });

  return (
    <section className="panel history-panel">
      <div className="panel-heading">
        <div>
          <h2>История загрузок</h2>
          <p>SQLite хранит результаты обработки и файлы отчетов.</p>
        </div>
        <button className="ghost-button" onClick={onExport}>
          <Download size={17} /> CSV
        </button>
      </div>
      <div className="history-tools">
        <label>
          <Search size={16} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Поиск по файлу, модели или объектам"
          />
        </label>
        <select value={profileFilter} onChange={(event) => setProfileFilter(event.target.value)}>
          <option value="all">Все модели</option>
          {profiles.map((profile) => (
            <option key={profile.id} value={profile.id}>{profile.short_title || profile.title}</option>
          ))}
        </select>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Дата</th>
              <th>Модель</th>
              <th>Файл</th>
              <th>Результат</th>
              <th>Время</th>
              <th>Действия</th>
            </tr>
          </thead>
          <tbody>
            {filteredHistory.map((row) => (
              <tr key={row.id}>
                <td>{new Date(row.created_at).toLocaleString('ru-RU')}</td>
                <td>{profileLabels[row.profile_id] || row.profile_id}</td>
                <td>{row.original_name}</td>
                <td>{formatCounts(row.counts ?? row.counts_json)}</td>
                <td>{row.elapsed_ms} мс</td>
                <td>
                  <div className="table-actions">
                    <button className="icon-button" title="Открыть результат" onClick={() => onOpen(row)}>
                      <Eye size={16} />
                    </button>
                    <button className="icon-button" title="Скачать отчет" onClick={() => onReport(row)}>
                      <FileText size={16} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
            {filteredHistory.length === 0 && (
              <tr>
                <td colSpan="6" className="muted-cell">История пока пуста.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PaymentModal({ onClose, onPayment, company }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="payment-modal" onClick={(event) => event.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        <CreditCard size={28} />
        <h2>Оформление подписки</h2>
        <p>{company}</p>
        <div className="bank-card">
          <span>AGROVISION PRO</span>
          <strong>**** **** **** 2026</strong>
          <small>ежемесячный доступ</small>
        </div>
        <button className="primary-button" onClick={onPayment}>Подтвердить оплату</button>
      </div>
    </div>
  );
}

function safeCounts(raw) {
  if (raw && typeof raw === 'object') return raw;
  try {
    const data = typeof raw === 'string' ? JSON.parse(raw) : raw;
    return data && typeof data === 'object' ? data : {};
  } catch {
    return {};
  }
}

function formatCounts(raw) {
  const data = safeCounts(raw);
  const entries = Object.entries(data);
  if (!entries.length) return 'нет объектов';
  return entries.map(([key, value]) => `${key}: ${value}`).join(', ');
}

createRoot(document.getElementById('root')).render(<App />);
