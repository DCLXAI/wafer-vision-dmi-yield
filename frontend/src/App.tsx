import { Suspense, lazy, useCallback, useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import { config, realClient } from './api';
import { mockClient, setMockPreview } from './mock';
import { createPattern, LABELS, labelColor } from './wafer';
import { ErrorBoundary } from './components/ErrorBoundary';
import { compact, dt, ms, pct } from './utils/format';
import { previewFile } from './parsers';
import { copy, labelInfo, labelName, labelOptions, LANG_STORAGE_KEY, normalizeLanguage, type Language } from './i18n';
import type { ApiClient, DefectLabel, FeatureResponse, HealthResponse, HistoryPage, ModelMetadata, PredictResponse, StatsSummary, WaferMatrix } from './types';
import './styles.css';

const SpotfireSimulator = lazy(() => import('./SpotfireSimulator').then((m) => ({ default: m.SpotfireSimulator })));
const DashboardCharts = lazy(() => import('./components/DashboardCharts').then((m) => ({ default: m.DashboardCharts })));
const cellColor = (v: number) => v >= 1.5 ? 'var(--defect-hot)' : v > 0 ? 'var(--die-good)' : 'transparent';
const clamp01 = (value: number) => Math.max(0, Math.min(1, value));
const client = (mode: 'real' | 'mock'): ApiClient => mode === 'mock' ? mockClient : realClient;

type MenuName = 'file' | 'edit' | 'view' | 'tools' | 'help' | 'settings' | null;
type SheetId = 'wafer' | 'simulator' | 'diagnostics' | 'history';
type WaferStats = { coverage: number; defectRatio: number; edgeSignal: number; inspected: number; defective: number };

export default function App() {
  const [language, setLanguageState] = useState<Language>(() => {
    const urlLanguage = new URLSearchParams(window.location.search).get('lang');
    if (urlLanguage === 'ko' || urlLanguage === 'en') return urlLanguage;
    const storedLanguage = localStorage.getItem(LANG_STORAGE_KEY);
    return normalizeLanguage(storedLanguage ?? (navigator.language?.startsWith('ko') ? 'ko' : 'en'));
  });
  const [mode, setMode] = useState<'real' | 'mock'>(config.useMocks ? 'mock' : 'real');
  const api = useMemo(() => client(mode), [mode]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [model, setModel] = useState<ModelMetadata | null>(null);
  const [stats, setStats] = useState<StatsSummary | null>(null);
  const [history, setHistory] = useState<HistoryPage | null>(null);
  const [filter, setFilter] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [sample, setSample] = useState<string | null>('Edge-Ring');
  const [matrix, setMatrix] = useState<WaferMatrix | null>(createPattern('Edge-Ring'));
  const [warning, setWarning] = useState<string | null>(null);
  const [note, setNote] = useState('');
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [features, setFeatures] = useState<FeatureResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [menu, setMenu] = useState<MenuName>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isEditing, setIsEditing] = useState(true);
  const [compactDensity, setCompactDensity] = useState(() => localStorage.getItem('wafervision.density') === 'compact');
  const [activeSheet, setActiveSheet] = useState<SheetId>('wafer');
  const [toast, setToast] = useState<string | null>(null);
  const c = copy[language];

  useEffect(() => {
    document.documentElement.lang = language;
    document.title = c.meta.title;
    setMeta('name', 'description', c.meta.description);
    setMeta('property', 'og:title', c.meta.title);
    setMeta('property', 'og:description', c.meta.description);
    setMeta('name', 'twitter:title', c.meta.title);
    setMeta('name', 'twitter:description', c.meta.description);
  }, [c.meta.description, c.meta.title, language]);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const [h, m, s, page] = await Promise.all([
        api.getHealth(), api.getModel(), api.getStats(), api.listPredictions({ limit: 50, label: filter || undefined }),
      ]);
      setHealth(h); setModel(m); setStats(s); setHistory(page);
    } catch (e) {
      const message = e instanceof Error ? e.message : 'API connection failed.';
      setError(message);
      if (mode === 'real') setHealth({ status: 'degraded', app: 'WaferVision API', environment: 'local', model_loaded: false, database: 'unavailable' });
    } finally {
      setRefreshing(false);
    }
  }, [api, filter, mode]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { setMockPreview(matrix); }, [matrix]);
  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 2400);
    return () => window.clearTimeout(timer);
  }, [toast]);

  function notify(message: string) { setToast(message); }

  function setLanguage(next: Language) {
    setLanguageState(next);
    localStorage.setItem(LANG_STORAGE_KEY, next);
    const url = new URL(window.location.href);
    url.searchParams.set('lang', next);
    window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`);
    notify(copy[next].toast.language);
  }

  function scrollToSection(id: string, sheet?: SheetId) {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (sheet) setActiveSheet(sheet);
    setMenu(null);
  }

  function resetWorkspaceView() {
    setFile(null);
    setSample('Edge-Ring');
    setMatrix(createPattern('Edge-Ring'));
    setPrediction(null);
    setFeatures(null);
    setWarning(null);
    setNote('');
    setSearchQuery('');
    scrollToSection('wafer-workspace', 'wafer');
    notify(c.toast.reset);
  }

  async function chooseFile(next: File) {
    setFile(next); setSample(null); setPrediction(null); setFeatures(null); setWarning(null);
    const preview = await previewFile(next);
    setMatrix(preview.matrix); setWarning(preview.warning ?? null);
  }

  function chooseSample(label: DefectLabel) {
    setFile(null); setSample(label); setPrediction(null); setFeatures(null); setWarning(null); setMatrix(createPattern(label));
  }

  async function analyze() {
    if (!file && !sample) {
      notify(c.toast.noRunTarget);
      return;
    }
    setLoading(true); setError(null);
    try {
      const activeMatrix = matrix ?? createPattern(sample ?? 'Random');
      const [result, featureResult] = file
        ? await Promise.all([api.predictFile(file, note || undefined), api.extractFeaturesFile(file)])
        : await Promise.all([
            api.predictArray(activeMatrix, `${sample}-synthetic.csv`, note || undefined),
            api.extractFeaturesArray(activeMatrix, `${sample}-synthetic.csv`),
          ]);
      setPrediction(result);
      setFeatures(featureResult);
      await refresh();
      scrollToSection('prediction-panel');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Inference failed.');
    } finally { setLoading(false); }
  }

  const historyItems = history?.items ?? stats?.latest ?? [];
  const visibleHistory = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return historyItems;
    return historyItems.filter((item) => [
      item.filename,
      item.input_kind,
      item.predicted_label,
      labelName(item.predicted_label, language),
      item.model_version,
      String(item.id),
    ].filter(Boolean).some((value) => String(value).toLowerCase().includes(query)));
  }, [historyItems, language, searchQuery]);
  const healthy = mode === 'mock' || health?.status === 'ok';
  const totalPredictions = stats?.total_predictions ?? 0;
  const avgConfidence = stats?.average_confidence ?? null;
  const macroF1 = model?.validation_macro_f1 ?? null;
  const featureDim = model?.feature_dim ?? features?.feature_dim ?? 59;
  const lastLatency = prediction?.inference_ms ?? historyItems[0]?.inference_ms;

  function exportVisibleHistory() {
    const header = ['id', 'created_at', 'filename', 'input_kind', 'predicted_label', 'confidence', 'inference_ms', 'rows', 'cols'] as const;
    const lines = [header.join(',')].concat(visibleHistory.map((item) => header.map((key) => csvEscape(String(item[key as keyof typeof item] ?? ''))).join(',')));
    downloadText(`wafervision-history-${Date.now()}.csv`, lines.join('\n'), 'text/csv;charset=utf-8');
    notify(c.toast.exported);
    setMenu(null);
  }

  function saveWorkspaceSettings() {
    localStorage.setItem(LANG_STORAGE_KEY, language);
    localStorage.setItem('wafervision.density', compactDensity ? 'compact' : 'comfortable');
    notify(c.toast.saved);
    setMenu(null);
  }

  async function copyPublicLink() {
    const url = new URL(window.location.href);
    url.searchParams.set('lang', language);
    try {
      await navigator.clipboard.writeText(url.toString());
      notify(c.toast.linkCopied);
    } catch {
      notify(c.toast.linkCopyFailed);
    }
    setMenu(null);
  }

  function toggleEditing() {
    setIsEditing((current) => {
      notify(current ? c.toast.editOff : c.toast.editOn);
      return !current;
    });
    setMenu(null);
  }

  function toggleDensity() {
    setCompactDensity((current) => {
      const next = !current;
      localStorage.setItem('wafervision.density', next ? 'compact' : 'comfortable');
      return next;
    });
    setMenu(null);
  }

  return (
    <>
    <a className="skip-link" href="#wafer-workspace">{c.meta.skip}</a>
    <div className={`spotfire-workbench ${compactDensity ? 'density-compact' : ''} ${isEditing ? 'editing-on' : 'editing-off'}`}>
      <aside className="side-rail" aria-label={c.chrome.layouts}>
        <button aria-label={c.chrome.addAnalysis} title={c.chrome.addAnalysis} onClick={() => scrollToSection('upload-panel', 'wafer')}>+</button>
        <button aria-label={c.chrome.dataTable} title={c.chrome.dataTable} onClick={() => scrollToSection('history-panel', 'history')}><span className="rail-table" /></button>
        <button aria-label={c.chrome.visualizations} title={c.chrome.visualizations} onClick={() => scrollToSection('charts-panel', 'diagnostics')}><span className="rail-bars" /></button>
        <button aria-label={c.chrome.expressions} title={c.chrome.expressions} onClick={() => scrollToSection('features-panel', 'diagnostics')}>f(x)</button>
        <button aria-label={c.chrome.layouts} title={c.chrome.layouts} onClick={() => scrollToSection('wafer-workspace', 'wafer')}><span className="rail-layout" /></button>
      </aside>
      <div className="workbench-main">
        <nav className="top-toolbar" aria-label={c.chrome.layouts}>
          <div className="toolbar-left">
            <button className="toolbar-icon" aria-label={c.chrome.more} onClick={() => setMenu(menu === 'file' ? null : 'file')}>...</button>
            <span className="toolbar-divider" />
            <button className={menu === 'file' ? 'active' : ''} onClick={() => setMenu(menu === 'file' ? null : 'file')}>{c.chrome.file}</button>
            <button className={menu === 'edit' ? 'active' : ''} onClick={() => setMenu(menu === 'edit' ? null : 'edit')}>{c.chrome.edit}</button>
            <button onClick={() => scrollToSection('history-panel', 'history')}>{c.chrome.data}</button>
            <button onClick={() => scrollToSection('charts-panel', 'diagnostics')}>{c.chrome.visualizations}</button>
            <button className={menu === 'view' ? 'active' : ''} onClick={() => setMenu(menu === 'view' ? null : 'view')}>{c.chrome.view}</button>
            <button className={menu === 'tools' ? 'active' : ''} onClick={() => setMenu(menu === 'tools' ? null : 'tools')}>{c.chrome.tools}</button>
            <button className={menu === 'help' ? 'active' : ''} onClick={() => setMenu(menu === 'help' ? null : 'help')}>{c.chrome.help}</button>
          </div>
          <div className="toolbar-right">
            <button className={`toolbar-icon ${searchOpen ? 'active' : ''}`} aria-label={c.chrome.search} onClick={() => { setSearchOpen((v) => !v); setMenu(null); }}>{c.chrome.search}</button>
            <button className="toolbar-icon" aria-label={c.chrome.filter} onClick={() => scrollToSection('simulator-filters', 'simulator')}>{c.chrome.filter}</button>
            <button className={`toolbar-icon ${menu === 'settings' ? 'active' : ''}`} aria-label={c.chrome.settings} onClick={() => setMenu(menu === 'settings' ? null : 'settings')}>{c.chrome.settings}</button>
            <select className="language-select" value={language} onChange={(e) => setLanguage(e.target.value as Language)} aria-label={c.chrome.language}>
              <option value="ko">{c.chrome.korean}</option>
              <option value="en">{c.chrome.english}</option>
            </select>
            <span className="toolbar-divider" />
            <button className="edit-mode" onClick={toggleEditing}>{isEditing ? c.chrome.editing : c.chrome.viewing}</button>
          </div>
        </nav>

        {menu && <ToolbarMenu menu={menu} language={language} compactDensity={compactDensity} onClose={() => setMenu(null)} onRefresh={() => { void refresh(); notify(c.toast.refreshed); setMenu(null); }} onExport={exportVisibleHistory} onCopyLink={() => void copyPublicLink()} onSave={saveWorkspaceSettings} onToggleEdit={toggleEditing} onReset={resetWorkspaceView} onTop={() => scrollToSection('wafer-workspace', 'wafer')} onDensity={toggleDensity} onRun={() => void analyze()} onSimulator={() => scrollToSection('simulator-filters', 'simulator')} onLanguage={setLanguage} />}
        {searchOpen && <SearchPanel language={language} query={searchQuery} count={visibleHistory.length} onQuery={setSearchQuery} onClose={() => setSearchOpen(false)} onHistory={() => scrollToSection('history-panel', 'history')} />}
        {toast && <div className="toast" role="status">{toast}</div>}

        <main className="app-shell">
          <header className="hero" id="wafer-workspace">
            <div className="status-bar">
              <div className="status-left">
                <span className={`status-dot ${healthy ? 'ok' : 'bad'}`} />
                <span>{mode === 'mock' ? c.hero.statusMock : health?.app ?? c.hero.statusReal}</span>
                <span className="separator" />
                <span>{mode === 'real' ? config.apiBaseUrl : c.hero.localInference}</span>
              </div>
              <div className="status-actions">
                <span className="model-badge">{model?.loaded ? model.model_version : c.hero.modelMissing}</span>
                <div className="toggle">
                  <button className={mode === 'real' ? 'active' : ''} onClick={() => { setMode('real'); setError(null); }}>{c.hero.realApi}</button>
                  <button className={mode === 'mock' ? 'active' : ''} onClick={() => { setMode('mock'); setError(null); }}>{c.hero.demo}</button>
                </div>
                <button className="ghost" onClick={() => void refresh()} disabled={refreshing}>{refreshing ? c.hero.refreshing : c.hero.refresh}</button>
              </div>
            </div>
            <DeploymentStrip language={language} mode={mode} />

            <div className="hero-content">
              <div>
                <p className="eyebrow">{c.hero.eyebrow}</p>
                <h1>{c.hero.title}</h1>
                <p className="hero-copy">{c.hero.copy}</p>
              </div>
              <HeroVisualCard language={language} prediction={prediction} stats={stats} />
            </div>

            <div className="metric-grid">
              <Kpi variant="volume" meter={Math.min(1, totalPredictions / 120)} label={c.kpi.total} value={compact(totalPredictions, language)} hint={mode === 'mock' ? c.kpi.totalHintMock : c.kpi.totalHintReal} />
              <Kpi variant="confidence" meter={avgConfidence ?? 0.78} label={c.kpi.confidence} value={pct(avgConfidence)} hint={c.kpi.confidenceHint} />
              <Kpi variant="model" meter={macroF1 ?? 0.79} label={c.kpi.macro} value={pct(macroF1)} hint={model?.loaded ? model.model_version : c.kpi.modelUnavailable} />
              <Kpi variant="feature" meter={Math.min(1, featureDim / 80)} label={c.kpi.featureDim} value={String(featureDim)} hint={model?.model_kind ?? c.kpi.featureHintFallback} />
              <Kpi variant="latency" meter={1 - Math.min(1, (lastLatency ?? 80) / 260)} label={c.kpi.latency} value={ms(lastLatency)} hint={healthy ? c.kpi.serviceHealthy : c.kpi.waitingBackend} />
            </div>
          </header>

          {error && <aside className="error"><strong>{mode === 'real' ? c.panels.backendIssue : c.panels.demoIssue}</strong><span>{error}</span>{mode === 'real' && <button onClick={() => setMode('mock')}>{c.panels.switchDemo}</button>}</aside>}

          <section className="workspace" id="workspace-grid">
            <UploadPanel language={language} file={file} sample={sample} warning={warning} note={note} loading={loading} disabled={!isEditing} onFile={chooseFile} onSample={chooseSample} onNote={setNote} onAnalyze={analyze} onReset={resetWorkspaceView} />
            <WaferHeatmap language={language} matrix={matrix} title={prediction ? `${labelName(prediction.label, language)} ${c.wafer.signature}` : c.wafer.preview} />
            <PredictionResult language={language} prediction={prediction} loading={loading} />
          </section>

          <FeaturePanel language={language} features={features} />

          <ErrorBoundary eyebrow={language === 'ko' ? 'UI 안전 경계' : 'UI safety boundary'} title={language === 'ko' ? '시뮬레이터 패널 보호됨' : 'Simulator cockpit failed safely'} retryLabel={language === 'ko' ? '패널 다시 시도' : 'Retry panel'}>
            <Suspense fallback={<section className="card"><p className="eyebrow">{c.panels.loadingSimulator}</p><h2>{c.panels.preparingSimulator}</h2></section>}>
              <SpotfireSimulator api={api} mode={mode} language={language} onError={setError} onMatrixSelect={(nextMatrix, label) => { setMatrix(nextMatrix); setSample(String(label)); setFile(null); }} />
            </Suspense>
          </ErrorBoundary>

          <Suspense fallback={<section className="charts"><div className="card chart wide skeleton-card"><p className="eyebrow">{c.panels.loadingCharts}</p><h2>{c.panels.preparingCharts}</h2></div></section>}>
            <DashboardCharts language={language} stats={stats} history={visibleHistory} />
          </Suspense>
          <History language={language} items={visibleHistory} total={history?.total ?? historyItems.length} filter={filter} setFilter={setFilter} />
          <footer className="sheet-tabs" aria-label={c.panels.sheetWafer}>
            <button className={activeSheet === 'wafer' ? 'active' : ''} onClick={() => scrollToSection('wafer-workspace', 'wafer')}>{c.panels.sheetWafer}</button>
            <button className={activeSheet === 'simulator' ? 'active' : ''} onClick={() => scrollToSection('simulator', 'simulator')}>{c.panels.sheetSimulator}</button>
            <button className={activeSheet === 'diagnostics' ? 'active' : ''} onClick={() => scrollToSection('charts-panel', 'diagnostics')}>{c.panels.sheetDiagnostics}</button>
            <button className={activeSheet === 'history' ? 'active' : ''} onClick={() => scrollToSection('history-panel', 'history')}>{c.panels.sheetHistory}</button>
            <span>{visibleHistory.length.toLocaleString(language === 'ko' ? 'ko-KR' : 'en-US')} / {Math.max(historyItems.length, history?.total ?? historyItems.length).toLocaleString(language === 'ko' ? 'ko-KR' : 'en-US')} {c.panels.rows}</span>
          </footer>
          <CreatorFooter />
        </main>
      </div>
    </div>
    </>
  );
}

function ToolbarMenu({ menu, language, compactDensity, onClose, onRefresh, onExport, onCopyLink, onSave, onToggleEdit, onReset, onTop, onDensity, onRun, onSimulator, onLanguage }: {
  menu: Exclude<MenuName, null>;
  language: Language;
  compactDensity: boolean;
  onClose(): void;
  onRefresh(): void;
  onExport(): void;
  onCopyLink(): void;
  onSave(): void;
  onToggleEdit(): void;
  onReset(): void;
  onTop(): void;
  onDensity(): void;
  onRun(): void;
  onSimulator(): void;
  onLanguage(next: Language): void;
}) {
  const c = copy[language];
  const title = {
    file: c.menus.fileTitle,
    edit: c.menus.editTitle,
    view: c.menus.viewTitle,
    tools: c.menus.toolsTitle,
    help: c.menus.helpTitle,
    settings: c.menus.settingsTitle,
  }[menu];
  return <section className="workbench-popover" role="dialog" aria-label={title}>
    <div className="popover-heading"><strong>{title}</strong><button onClick={onClose}>{c.chrome.close}</button></div>
    {menu === 'file' && <div className="popover-actions"><button onClick={onExport}>{c.menus.exportHistory}</button><button onClick={onCopyLink}>{c.menus.copyLink}</button><button onClick={onRefresh}>{c.menus.refreshData}</button><button onClick={onSave}>{c.menus.saveWorkspace}</button></div>}
    {menu === 'edit' && <div className="popover-actions"><button onClick={onToggleEdit}>{c.menus.toggleEdit}</button><button onClick={onReset}>{c.menus.resetWorkspace}</button></div>}
    {menu === 'view' && <div className="popover-actions"><button onClick={onTop}>{c.menus.top}</button><button onClick={onDensity}>{compactDensity ? c.chrome.comfortable : c.chrome.compact}</button></div>}
    {menu === 'tools' && <div className="popover-actions"><button onClick={onRun}>{c.menus.runClassification}</button><button onClick={onSimulator}>{c.menus.openSimulator}</button></div>}
    {menu === 'help' && <p className="popover-copy">{c.menus.helpBody}</p>}
    {menu === 'settings' && <div className="settings-grid"><label>{c.chrome.language}<select value={language} onChange={(e) => onLanguage(e.target.value as Language)}><option value="ko">{c.chrome.korean}</option><option value="en">{c.chrome.english}</option></select></label><p><span>{c.chrome.density}</span><strong>{compactDensity ? c.chrome.compact : c.chrome.comfortable}</strong></p></div>}
  </section>;
}

function SearchPanel({ language, query, count, onQuery, onClose, onHistory }: { language: Language; query: string; count: number; onQuery(v: string): void; onClose(): void; onHistory(): void }) {
  const c = copy[language];
  return <section className="toolbar-search" role="search">
    <label>{c.menus.searchTitle}<input value={query} onChange={(e) => onQuery(e.target.value)} placeholder={c.menus.searchPlaceholder} autoFocus /></label>
    <button onClick={onHistory}>{count.toLocaleString(language === 'ko' ? 'ko-KR' : 'en-US')}</button>
    <button onClick={onClose}>{c.chrome.close}</button>
    {!query && <p>{c.menus.noSearch}</p>}
  </section>;
}

function DeploymentStrip({ language, mode }: { language: Language; mode: 'real' | 'mock' }) {
  const c = copy[language];
  const apiIsLocal = /\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0)(:|\/|$)/.test(config.apiBaseUrl);
  const publicHost = !['localhost', '127.0.0.1', '0.0.0.0'].includes(window.location.hostname);
  const warning = publicHost && mode === 'real' && apiIsLocal;
  return <div className={`deployment-strip ${warning ? 'warning' : ''}`} role={warning ? 'alert' : 'status'}>
    <span>{warning ? c.deploy.apiWarning : mode === 'mock' ? c.deploy.demoReady : c.deploy.realReady}</span>
    <strong>{c.deploy.privacy}</strong>
  </div>;
}

function HeroVisualCard({ language, prediction, stats }: { language: Language; prediction: PredictResponse | null; stats: StatsSummary | null }) {
  const c = copy[language];
  const activeLabel = prediction?.label ?? stats?.label_counts?.[0]?.label ?? 'Edge-Ring';
  const activeColor = labelColor(activeLabel);
  const confidence = prediction?.confidence ?? stats?.average_confidence ?? 0.779;
  const confidenceOffset = 151 - Math.round(Math.max(0.05, Math.min(1, confidence)) * 151);
  const total = stats?.total_predictions ?? 29;
  return <aside className="hero-card hero-visual-card" style={{ '--active-defect': activeColor } as CSSProperties} aria-label={c.hero.buildStrong}>
    <div className="hero-visual-copy">
      <span>{c.hero.build}</span>
      <strong>{c.hero.buildStrong}</strong>
      <p>{c.hero.buildCopy}</p>
    </div>
    <svg className="hero-visual-scene" viewBox="0 0 360 220" role="img" aria-label={c.hero.buildStrong}>
      <defs>
        <linearGradient id="heroWaferShade" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#edf6ff" />
          <stop offset=".72" stopColor="#bfd7ee" />
          <stop offset="1" stopColor="#f0a34a" />
        </linearGradient>
        <linearGradient id="heroSignalBlue" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#4f79e7" />
          <stop offset=".55" stopColor="#35aeea" />
          <stop offset="1" stopColor="#36ec83" />
        </linearGradient>
        <filter id="heroSoftShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="8" stdDeviation="8" floodColor="#344660" floodOpacity=".18" />
        </filter>
      </defs>
      <rect x="4" y="6" width="352" height="208" rx="6" fill="#f8fbff" stroke="#cbd7e8" />
      <g className="hero-wafer-cluster" filter="url(#heroSoftShadow)">
        <circle cx="96" cy="104" r="62" fill="url(#heroWaferShade)" stroke="#c8792f" strokeWidth="2.4" />
        <circle cx="96" cy="104" r="49" fill="none" stroke="#8fa1b7" strokeOpacity=".34" strokeWidth="1.1" />
        <circle cx="96" cy="104" r="34" fill="none" stroke="#8fa1b7" strokeOpacity=".26" strokeWidth="1.1" />
        {Array.from({ length: 8 }).map((_, row) => Array.from({ length: 8 }).map((__, col) => {
          const x = 60 + col * 10;
          const y = 68 + row * 10;
          const dx = x + 4 - 96;
          const dy = y + 4 - 104;
          if (Math.sqrt(dx * dx + dy * dy) > 54) return null;
          const hot = (row === 1 && col > 4) || (row === 2 && col > 4) || (row === 5 && col < 3) || (row === 6 && col < 3) || (row === col && row > 2);
          return <rect key={`${row}-${col}`} x={x} y={y} width="7" height="7" rx="1.2" fill={hot ? activeColor : '#8bb6df'} opacity={hot ? .92 : .68} />;
        }))}
        <path d="M45 105c22 22 70 42 103 9" fill="none" stroke="#ffffff" strokeOpacity=".36" strokeWidth="7" strokeLinecap="round" />
      </g>
      <g className="hero-pipeline">
        <rect x="184" y="42" width="126" height="24" rx="4" fill="#ffffff" stroke="#cbd7e8" />
        <rect x="184" y="84" width="126" height="24" rx="4" fill="#ffffff" stroke="#cbd7e8" />
        <rect x="184" y="126" width="126" height="24" rx="4" fill="#ffffff" stroke="#cbd7e8" />
        <path d="M167 104h17M247 66v18M247 108v18" stroke="#94abd0" strokeWidth="1.5" strokeLinecap="round" />
        <rect x="196" y="50" width="54" height="7" rx="2" fill="#4f79e7" opacity=".78" />
        <rect x="196" y="92" width="74" height="7" rx="2" fill="#36ec83" opacity=".78" />
        <rect x="196" y="134" width="42" height="7" rx="2" fill="#ffd84d" opacity=".9" />
        <circle cx="288" cy="54" r="6" fill="#4f79e7" opacity=".78" />
        <circle cx="288" cy="96" r="6" fill="#36ec83" opacity=".78" />
        <circle cx="288" cy="138" r="6" fill="#e84d6f" opacity=".82" />
      </g>
      <g className="hero-sparkline">
        <path d="M48 188h268" stroke="#d6deeb" strokeWidth="1" />
        <path className="hero-signal-path" d="M48 176c22-18 39 12 61-2s33-34 56-18 39 30 66 4 48-31 85-8" fill="none" stroke="url(#heroSignalBlue)" strokeWidth="3.2" strokeLinecap="round" />
        <path d="M48 190c32-3 55-14 82-8s45 17 76 10 49-20 110-12" fill="none" stroke="#e84d6f" strokeOpacity=".72" strokeWidth="2.4" strokeLinecap="round" />
      </g>
      <g className="hero-confidence-ring" transform="translate(320 36)">
        <circle cx="0" cy="0" r="24" fill="#ffffff" stroke="#d6deeb" strokeWidth="1.2" />
        <circle cx="0" cy="0" r="24" fill="none" stroke="#36ec83" strokeWidth="4" strokeDasharray="151" strokeDashoffset={confidenceOffset} strokeLinecap="round" transform="rotate(-90)" />
        <circle cx="0" cy="0" r="6" fill={activeColor} />
      </g>
      <g className="hero-run-counters">
        <rect x="242" y="171" width="74" height="26" rx="4" fill="#ffffff" stroke="#cbd7e8" />
        <rect x="251" y="180" width={Math.max(14, Math.min(54, total % 100))} height="7" rx="2" fill="#4f79e7" opacity=".78" />
      </g>
    </svg>
  </aside>;
}

function CreatorFooter() {
  return <footer className="creator-footer" aria-label="creator contact">
    <span>제작자 <strong>정순수</strong></span>
    <a href="mailto:jss5797@naver.com">jss5797@naver.com</a>
    <a className="social-link instagram-link" href="https://www.instagram.com/sunsunox/" target="_blank" rel="noreferrer" aria-label="Instagram sunsunox">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="3.5" y="3.5" width="17" height="17" rx="5" />
        <circle cx="12" cy="12" r="4.2" />
        <circle cx="17.2" cy="6.8" r="1.1" />
      </svg>
    </a>
    <a className="social-link threads-link" href="https://www.threads.com/@sunsunox" target="_blank" rel="noreferrer" aria-label="Threads sunsunox">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12.2 3.2c4.6 0 7.5 2.9 7.5 8.8 0 5.4-3.1 8.8-7.8 8.8-4.6 0-7.6-3.2-7.6-8.5 0-5.4 3-9.1 7.9-9.1Z" />
        <path d="M15.8 9.8c-.5-1.8-1.7-2.8-3.6-2.8-2.3 0-3.8 1.7-3.8 4.7 0 3.2 1.4 5.2 3.8 5.2 2 0 3.4-1.1 3.4-2.7 0-1.3-1-2.1-2.8-2.1h-1.1" />
        <path d="M17.5 12.1c-1.3-.8-3-1.2-5.1-1.1" />
      </svg>
    </a>
    <a className="social-link youtube-link" href="https://www.youtube.com/channel/UCHD0-T7C_F6o5MYUHsVIUnQ" target="_blank" rel="noreferrer" aria-label="YouTube channel">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="3" y="6.2" width="18" height="11.6" rx="3.2" />
        <path d="M10.3 9.2v5.6l5-2.8-5-2.8Z" />
      </svg>
    </a>
  </footer>;
}

type KpiVariant = 'volume' | 'confidence' | 'model' | 'feature' | 'latency';

function Kpi({ label, value, hint, meter, variant }: { label: string; value: string; hint: string; meter: number; variant: KpiVariant }) {
  const safeMeter = Math.max(0.04, Math.min(1, Number.isFinite(meter) ? meter : 0.5));
  return <section className={`kpi kpi-${variant}`} style={{ '--meter': safeMeter } as CSSProperties}>
    <div className="kpi-copy"><span>{label}</span><strong>{value}</strong><p>{hint}</p></div>
    <KpiVisual variant={variant} meter={safeMeter} />
  </section>;
}

function KpiVisual({ variant, meter }: { variant: KpiVariant; meter: number }) {
  const bars = Array.from({ length: 7 }, (_, index) => {
    const wave = variant === 'latency' ? 7 - index : index + 1;
    return Math.max(5, Math.round(5 + wave * 2.2 + meter * (index % 3 === 0 ? 12 : 7)));
  });
  const ringOffset = 63 - meter * 63;
  return <svg className="kpi-visual" viewBox="0 0 104 48" aria-hidden="true">
    <rect x="1" y="1" width="102" height="46" rx="4" className="kpi-visual-shell" />
    <g className="kpi-bars">
      {bars.map((height, index) => <rect key={index} x={8 + index * 8} y={38 - height} width="4.5" height={height} rx="1.4" />)}
    </g>
    <path className="kpi-trace" d="M8 34c12-18 20 6 31-7s18-22 30-8 18 15 28-2" />
    <circle className="kpi-ring-bg" cx="82" cy="24" r="10" />
    <circle className="kpi-ring" cx="82" cy="24" r="10" strokeDasharray="63" strokeDashoffset={ringOffset} />
    <circle className="kpi-node" cx="82" cy="24" r="3.2" />
  </svg>;
}

function UploadPanel({ language, file, sample, warning, note, loading, disabled, onFile, onSample, onNote, onAnalyze, onReset }: {
  language: Language; file: File | null; sample: string | null; warning: string | null; note: string; loading: boolean; disabled: boolean; onFile(file: File): void; onSample(label: DefectLabel): void; onNote(note: string): void; onAnalyze(): void; onReset(): void;
}) {
  const [drag, setDrag] = useState(false);
  const c = copy[language];
  const pick = (files: FileList | null) => { const f = files?.[0]; if (f) void onFile(f); };
  return <section className="card upload-card" id="upload-panel">
    <div className="section-heading"><div><p className="eyebrow">{c.upload.eyebrow}</p><h2>{c.upload.title}</h2></div><button className="ghost" onClick={onReset}>{c.upload.reset}</button></div>
    <label className={`dropzone ${drag ? 'active' : ''}`} onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files); }}>
      <input type="file" accept=".csv,.npy,.png,.jpg,.jpeg,.bmp,.webp,image/*" hidden disabled={disabled} onChange={(e) => pick(e.target.files)} />
      <span className="drop-icon">⌁</span><strong>{file ? file.name : sample ? `${labelName(sample, language)} ${c.upload.synthetic}` : c.upload.drop}</strong><em>{c.upload.formats}</em>
    </label>
    {warning && <p className="warning">{warning}</p>}
    <div className="samples">{labelOptions(language).filter((l) => l.value !== 'None').slice(0, 8).map((item) => <button key={item.value} disabled={disabled} className={sample === item.value ? 'active' : ''} onClick={() => onSample(item.value)}>{item.label}</button>)}</div>
    <label className="note-label" htmlFor="note">{c.upload.note}</label>
    <textarea id="note" value={note} disabled={disabled} onChange={(e) => onNote(e.target.value)} placeholder={c.upload.notePlaceholder} />
    <button className="primary" onClick={onAnalyze} disabled={loading || (!file && !sample)}>{loading ? c.upload.analyzing : c.upload.run}</button>
  </section>;
}

function computeWaferStats(matrix: WaferMatrix): WaferStats {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const cx = cols / 2;
  const cy = rows / 2;
  const radius = Math.min(cols, rows) / 2;
  const activeArea = Math.PI * (radius * 0.96) ** 2;
  let inspected = 0;
  let defective = 0;
  let edgeDefective = 0;

  matrix.forEach((row, y) => row.forEach((value, x) => {
    if (value <= 0) return;
    inspected += 1;
    if (value < 1.5) return;
    defective += 1;
    const distance = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
    if (distance > radius * 0.72) edgeDefective += 1;
  }));

  return {
    coverage: clamp01(activeArea ? inspected / activeArea : 0),
    defectRatio: inspected ? defective / inspected : 0,
    edgeSignal: defective ? edgeDefective / defective : 0,
    inspected,
    defective,
  };
}

function WaferHeatmap({ language, matrix, title }: { language: Language; matrix: WaferMatrix | null; title: string }) {
  const c = copy[language];
  const rows = matrix?.length ?? 0;
  const cols = matrix?.[0]?.length ?? 0;
  const hasMap = Boolean(matrix && rows && cols);
  const cx = cols / 2;
  const cy = rows / 2;
  const radius = Math.min(cols, rows) * 0.48;
  const clipId = `wafer-clip-${rows}-${cols}`;
  const glowId = `wafer-glow-${rows}-${cols}`;
  const stats = hasMap && matrix ? computeWaferStats(matrix) : null;
  const hotspots = matrix
    ? matrix.flatMap((row, y) => row.map((value, x) => ({ value, x, y }))).filter((cell) => cell.value >= 1.5)
    : [];
  const hotspotStep = Math.max(1, Math.floor(hotspots.length / 26));
  const visibleHotspots = hotspots.filter((_, index) => index % hotspotStep === 0).slice(0, 26);
  const metrics = stats ? [
    { label: c.wafer.coverage, value: stats.coverage, accent: 'coverage' },
    { label: c.wafer.defectRatio, value: stats.defectRatio, accent: 'defect' },
    { label: c.wafer.edgeSignal, value: stats.edgeSignal, accent: 'edge' },
  ] : [];

  return <section className="card wafer-panel" id="wafer-map-panel">
    <div className="section-heading">
      <div><p className="eyebrow">{c.wafer.eyebrow}</p><h2>{title}</h2></div>
      <span className="dim">{rows && cols ? `${rows}×${cols}` : c.wafer.noMap}</span>
    </div>
    <div className="wafer-frame enhanced-wafer-frame">
      {hasMap && matrix ? <div className="wafer-stage">
        <svg viewBox={`0 0 ${cols} ${rows}`} className="wafer-svg" role="img" aria-label={title}>
          <defs>
            <clipPath id={clipId}><circle cx={cx} cy={cy} r={radius} /></clipPath>
            <radialGradient id={glowId} cx="50%" cy="48%" r="54%">
              <stop offset="0%" stopColor="#ffffff" stopOpacity="0.86" />
              <stop offset="68%" stopColor="#d7e7f8" stopOpacity="0.56" />
              <stop offset="100%" stopColor="#a9bed8" stopOpacity="0.74" />
            </radialGradient>
          </defs>
          <rect width={cols} height={rows} className="wafer-grid-base" />
          <circle cx={cx} cy={cy} r={radius} fill={`url(#${glowId})`} className="wafer-disk" />
          <g clipPath={`url(#${clipId})`}>
            <rect width={cols} height={rows} className="wafer-map-backdrop" />
            {matrix.map((row, y) => row.map((v, x) => v <= 0 ? null : <rect key={`${x}-${y}`} className={`wafer-cell ${v >= 1.5 ? 'defect' : 'good'}`} x={x} y={y} width={0.92} height={0.92} rx={0.12} fill={cellColor(v)} opacity={v >= 1.5 ? 0.96 : 0.62} />))}
            <line x1={cols * 0.02} y1={rows * 0.24} x2={cols * 0.98} y2={rows * 0.24} className="wafer-scan-line" />
          </g>
          <g className="wafer-hotspots">
            {visibleHotspots.map((cell) => <circle key={`${cell.x}-${cell.y}`} cx={cell.x + 0.46} cy={cell.y + 0.46} r={0.58} />)}
          </g>
          <g className="wafer-overlays">
            {[0.22, 0.44, 0.66, 0.88].map((scale) => <circle key={scale} cx={cx} cy={cy} r={radius * scale} />)}
            <line x1={cx - radius} y1={cy} x2={cx + radius} y2={cy} />
            <line x1={cx} y1={cy - radius} x2={cx} y2={cy + radius} />
          </g>
        </svg>
        <div className="wafer-scan-badge"><span />{c.wafer.scan}</div>
        {stats && <div className="wafer-coordinate-chip">{compact(stats.inspected, language)} {c.wafer.inspected}</div>}
      </div> : <div className="empty-wafer"><div /><p>{c.wafer.empty}</p></div>}
    </div>
    {stats && <div className="wafer-stat-rail">
      {metrics.map((metric) => <div key={metric.label} className={`wafer-stat ${metric.accent}`} style={{ '--stat': `${Math.round(clamp01(metric.value) * 100)}%` } as CSSProperties}>
        <span>{metric.label}</span><strong>{pct(metric.value, 0)}</strong><b><i /></b>
      </div>)}
    </div>}
    <div className="legend"><span><i className="good" />{c.wafer.good}</span><span><i className="bad" />{c.wafer.bad}</span><span><i />{c.wafer.outside}</span></div>
  </section>;
}

function PredictionResult({ language, prediction, loading }: { language: Language; prediction: PredictResponse | null; loading: boolean }) {
  const c = copy[language];
  if (loading) return <section className="card result loading" id="prediction-panel"><div className="skeleton title" /><div className="skeleton line" /><div className="skeleton line short" /><div className="spinner" /></section>;
  if (!prediction) return <section className="card result" id="prediction-panel"><p className="eyebrow">{c.prediction.eyebrow}</p><h2>{c.prediction.emptyTitle}</h2><p className="muted">{c.prediction.emptyBody}</p></section>;
  const info = labelInfo(prediction.label, language);
  const ringStyle = { '--confidence': `${prediction.confidence * 360}deg` } as CSSProperties;
  return <section className="card result" id="prediction-panel"><div className="result-top"><div><p className="eyebrow">{c.prediction.eyebrow} #{prediction.id}</p><h2 style={{ color: labelColor(prediction.label) }}>{labelName(prediction.label, language)}</h2></div><div className="ring" style={ringStyle}><span>{pct(prediction.confidence, 0)}</span></div></div><div className="meta"><div><span>{c.prediction.latency}</span><strong>{ms(prediction.inference_ms)}</strong></div><div><span>{c.prediction.created}</span><strong>{dt(prediction.created_at, language)}</strong></div><div><span>{c.prediction.model}</span><strong>{prediction.model.model_version}</strong></div></div><div className="topk">{prediction.top_k.map((item) => <div className="topk-row" key={item.label}><span><i style={{ background: labelColor(item.label) }} />{labelName(item.label, language)}</span><div><b style={{ width: `${item.probability * 100}%`, background: labelColor(item.label) }} /></div><strong>{pct(item.probability)}</strong></div>)}</div><article className="explain"><h3>{info.title}</h3><p>{info.cause}</p><p><strong>{c.prediction.nextCheck}</strong> {info.action}</p></article></section>;
}

function FeaturePanel({ language, features }: { language: Language; features: FeatureResponse | null }) {
  const c = copy[language];
  if (!features) return <section className="card feature-panel" id="features-panel"><div className="section-heading"><div><p className="eyebrow">{c.features.eyebrow}</p><h2>{c.features.emptyTitle}</h2></div><span className="dim">13 + 40 + 6</span></div><p className="muted">{c.features.emptyBody}</p></section>;
  const region = features.groups.find((g) => g.name === 'region_density');
  const geometry = features.groups.find((g) => g.name === 'geometry');
  const regionValues = region?.values ?? [];
  const geometryValues = geometry?.values ?? [];
  const geometryLabels = geometry?.labels ?? [];
  const maxRegion = Math.max(1, ...regionValues, 1);
  return <section className="card feature-panel" id="features-panel"><div className="section-heading"><div><p className="eyebrow">{c.features.eyebrow}</p><h2>{c.features.title}</h2></div><span className="dim">{features.feature_dim} {c.features.features}</span></div><div className="feature-grid"><div><h3>{c.features.region}</h3><div className="mini-bars">{regionValues.map((v, i) => <div key={i} className="mini-row"><span>{i + 1}</span><b style={{ width: `${Math.max(2, (v / maxRegion) * 100)}%` }} /><em>{v.toFixed(2)}</em></div>)}</div></div><div><h3>{c.features.geometry}</h3><div className="geometry-list">{geometryValues.map((v, i) => { const label = geometryLabels[i] ?? `geometry_${i}`; return <p key={label}><span>{label.replace('geom_', '').replace('_norm', '')}</span><strong>{v.toFixed(4)}</strong></p>; })}</div></div></div><p className="footnote">{c.features.footnote}</p></section>;
}

function History({ language, items, total, filter, setFilter }: { language: Language; items: Array<{ id: number; created_at: string; filename: string | null; input_kind: string; predicted_label: string; confidence: number; inference_ms: number; rows: number | null; cols: number | null }>; total: number; filter: string; setFilter(v: string): void }) {
  const c = copy[language];
  return <section className="card history" id="history-panel"><div className="section-heading"><div><p className="eyebrow">{c.history.eyebrow}</p><h2>{c.history.title}</h2></div><select value={filter} onChange={(e) => setFilter(e.target.value)}><option value="">{c.history.allLabels}</option>{labelOptions(language).map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></div><div className="table-wrap"><table><thead><tr><th>{c.history.id}</th><th>{c.history.created}</th><th>{c.history.file}</th><th>{c.history.prediction}</th><th>{c.history.confidence}</th><th>{c.history.latency}</th><th>{c.history.shape}</th></tr></thead><tbody>{items.map((item) => { const style = { '--chip': labelColor(item.predicted_label) } as CSSProperties; return <tr key={item.id}><td>#{item.id}</td><td>{dt(item.created_at, language)}</td><td className="filename">{item.filename ?? item.input_kind}</td><td><span className="chip" style={style}>{labelName(item.predicted_label, language)}</span></td><td>{pct(item.confidence)}</td><td>{ms(item.inference_ms)}</td><td>{item.rows && item.cols ? `${item.rows}×${item.cols}` : '—'}</td></tr>; })}{!items.length && <tr><td colSpan={7} className="empty">{c.history.empty}</td></tr>}</tbody></table></div><p className="footnote">{c.history.showing} {items.length.toLocaleString(language === 'ko' ? 'ko-KR' : 'en-US')} {c.history.of} {total.toLocaleString(language === 'ko' ? 'ko-KR' : 'en-US')} {c.history.records}{filter ? ` ${c.history.forLabel} ${labelName(filter, language)}` : ''}.</p></section>;
}

function csvEscape(value: string) {
  return /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

function downloadText(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function setMeta(attribute: 'name' | 'property', key: string, value: string) {
  let element = document.head.querySelector<HTMLMetaElement>(`meta[${attribute}="${key}"]`);
  if (!element) {
    element = document.createElement('meta');
    element.setAttribute(attribute, key);
    document.head.appendChild(element);
  }
  element.content = value;
}
