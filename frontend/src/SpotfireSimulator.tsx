import { Suspense, lazy, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import type { ApiClient, DefectLabel, SimulatorPerformanceMode, SimulatorRequest, SimulatorResponse, WaferMatrix } from './types';
import { LABELS, labelColor } from './wafer';
import { ConfusionLegend, ConfusionMatrix, MiniKpi, Slider, WaferCanvas, severityClass } from './components/SimulatorWidgets';
import { exportRootCausesCsv, exportSummaryJson, exportWafersCsv } from './utils/export';
import { dt, ms, pct, pp, risk } from './utils/format';
import { useSimulatorSessions } from './hooks/useSimulatorSessions';
import { copy, causeHintText, evidenceText, labelInfo, labelName, persistenceText, processName, recommendationText, scenarioText, severityName, type Language } from './i18n';

const PatternDistributionChart = lazy(() => import('./components/SimulatorCharts').then((m) => ({ default: m.PatternDistributionChart })));
const SimulatorAnalyticsCharts = lazy(() => import('./components/SimulatorCharts').then((m) => ({ default: m.SimulatorAnalyticsCharts })));

const SCENARIOS: Record<string, { mix: Record<string, number>; noise: number; density: number; mixed: number }> = {
  'balanced-baseline': {
    mix: { 'Edge-Ring': 0.2, Loc: 0.16, Random: 0.16, 'Edge-Loc': 0.13, Center: 0.1, Scratch: 0.1, Donut: 0.07, 'Near-full': 0.04, None: 0.04 },
    noise: 0.045,
    density: 1.0,
    mixed: 0.08,
  },
  'edge-ring-excursion': {
    mix: { 'Edge-Ring': 0.52, 'Edge-Loc': 0.18, Donut: 0.08, Loc: 0.08, Random: 0.06, Scratch: 0.03, Center: 0.03, None: 0.02 },
    noise: 0.035,
    density: 1.18,
    mixed: 0.12,
  },
  'scratch-handling-event': {
    mix: { Scratch: 0.46, Loc: 0.16, 'Edge-Loc': 0.10, Random: 0.12, 'Edge-Ring': 0.06, Center: 0.04, Donut: 0.03, None: 0.03 },
    noise: 0.055,
    density: 1.05,
    mixed: 0.16,
  },
  'noisy-mixed-lot': {
    mix: { Loc: 0.22, Random: 0.20, 'Edge-Loc': 0.16, Scratch: 0.12, 'Edge-Ring': 0.12, Center: 0.08, Donut: 0.05, 'Near-full': 0.03, None: 0.02 },
    noise: 0.09,
    density: 1.32,
    mixed: 0.28,
  },
  'critical-nearfull': {
    mix: { 'Near-full': 0.34, Random: 0.2, Loc: 0.14, 'Edge-Ring': 0.12, Scratch: 0.08, Center: 0.06, 'Edge-Loc': 0.04, None: 0.02 },
    noise: 0.075,
    density: 1.52,
    mixed: 0.18,
  },
};

interface Props {
  api: ApiClient;
  mode: 'real' | 'mock';
  language: Language;
  onMatrixSelect?: (matrix: WaferMatrix, label: DefectLabel) => void;
  onError?: (message: string) => void;
}

export function SpotfireSimulator({ api, mode, language, onMatrixSelect, onError }: Props) {
  const [scenario, setScenario] = useState('edge-ring-excursion');
  const [waferCount, setWaferCount] = useState(96);
  const [size, setSize] = useState(80);
  const [seed, setSeed] = useState(20260617);
  const [noise, setNoise] = useState(SCENARIOS['edge-ring-excursion'].noise);
  const [density, setDensity] = useState(SCENARIOS['edge-ring-excursion'].density);
  const [mixed, setMixed] = useState(SCENARIOS['edge-ring-excursion'].mixed);
  const [performanceMode, setPerformanceMode] = useState<SimulatorPerformanceMode>('balanced');
  const [useModel, setUseModel] = useState(true);
  const [result, setResult] = useState<SimulatorResponse | null>(null);
  const sessionPage = useSimulatorSessions(api, 8);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [labelFilter, setLabelFilter] = useState('');
  const [toolFilter, setToolFilter] = useState('');
  const [riskFilter, setRiskFilter] = useState('all');
  const [renderLimit, setRenderLimit] = useState(240);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const c = copy[language].simulator;
  const commonRunError = language === 'ko' ? '시뮬레이터 실행에 실패했습니다.' : 'Simulator run failed.';
  const commonSaveError = language === 'ko' ? '시뮬레이션 세션을 저장하지 못했습니다.' : 'Could not save simulation session.';
  const commonLoadError = language === 'ko' ? '시뮬레이션 세션을 불러오지 못했습니다.' : 'Could not load simulation session.';

  const activeScenario = SCENARIOS[scenario] ?? SCENARIOS['balanced-baseline'];
  const activeScenarioText = scenarioText(scenario, language);
  const selected = useMemo(() => result?.wafers.find((w) => w.id === selectedId) ?? result?.wafers[0] ?? null, [result, selectedId]);

  function applyScenario(next: string) {
    const s = SCENARIOS[next] ?? SCENARIOS['balanced-baseline'];
    setScenario(next);
    setNoise(s.noise);
    setDensity(s.density);
    setMixed(s.mixed);
  }

  async function run() {
    const payload: SimulatorRequest = {
      wafer_count: waferCount,
      size,
      seed,
      lot_count: Math.max(4, Math.min(16, Math.round(waferCount / 32))),
      noise_level: noise,
      defect_density_scale: density,
      mixed_pattern_rate: mixed,
      pattern_mix: activeScenario.mix,
      persist: false,
      scenario_name: scenario,
      use_model: useModel,
      model_batch_size: performanceMode === 'fidelity' ? 64 : 128,
      performance_mode: performanceMode,
      return_matrix_size: performanceMode === 'turbo' ? 40 : performanceMode === 'fidelity' ? Math.min(size, 96) : 56,
      downsample_method: performanceMode === 'fidelity' ? 'area' : 'nearest',
      severity_thresholds: { monitor: 24, warning: 45, critical: 72 },
    };
    setLoading(true);
    try {
      const response = waferCount >= 700 && api.startSimulatorJob && api.getSimulatorJob
        ? await runViaJob(api, payload, language)
        : await api.runSimulator(payload);
      setResult(response);
      const initial = response.wafers.find((w) => w.severity === 'Critical') ?? response.wafers.find((w) => w.risk_score >= 45) ?? response.wafers[0] ?? null;
      setSelectedId(initial?.id ?? null);
      if (initial) onMatrixSelect?.(initial.matrix, initial.predicted_label);
      if (response.persisted_wafer_count) await sessionPage.refresh(0);
    } catch (e) {
      onError?.(e instanceof Error ? e.message : commonRunError);
    } finally {
      setLoading(false);
    }
  }

  async function saveCurrentSession() {
    if (!result) return;
    setSaving(true);
    try {
      const saved = await api.saveSimulationSession(compactResponseForSave(result));
      setResult(saved);
      await sessionPage.refresh(0);
    } catch (e) {
      onError?.(e instanceof Error ? e.message : commonSaveError);
    } finally {
      setSaving(false);
    }
  }

  async function loadSession(sessionId: string) {
    setLoading(true);
    try {
      const response = await api.getSimulationSession(sessionId);
      setResult(response);
      setScenario(response.params.scenario_name ?? 'balanced-baseline');
      setWaferCount(response.params.wafer_count);
      setSize(response.params.size);
      setNoise(response.params.noise_level);
      setDensity(response.params.defect_density_scale);
      setMixed(response.params.mixed_pattern_rate);
      setSeed(response.params.seed ?? 20260617);
      setPerformanceMode(response.params.performance_mode ?? 'balanced');
      setUseModel(response.params.use_model ?? true);
      const first = response.wafers[0] ?? null;
      setSelectedId(first?.id ?? null);
      if (first) onMatrixSelect?.(first.matrix, first.predicted_label);
    } catch (e) {
      onError?.(e instanceof Error ? e.message : commonLoadError);
    } finally {
      setLoading(false);
    }
  }

  const visibleWafers = useMemo(() => {
    return (result?.wafers ?? []).filter((wafer) => {
      if (labelFilter && wafer.predicted_label !== labelFilter) return false;
      if (toolFilter && wafer.tool_id !== toolFilter) return false;
      if (riskFilter === 'critical' && wafer.severity !== 'Critical') return false;
      if (riskFilter === 'high' && wafer.risk_score < 45) return false;
      if (riskFilter === 'low' && wafer.risk_score >= 45) return false;
      return true;
    });
  }, [result, labelFilter, toolFilter, riskFilter]);

  const tools = useMemo(() => [...new Set((result?.wafers ?? []).map((w) => w.tool_id))].sort(), [result]);
  const scatter = useMemo(() => visibleWafers.map((w) => ({ x: Number((w.defect_density * 100).toFixed(2)), y: Number((w.confidence * 100).toFixed(1)), z: w.risk_score, label: w.predicted_label, id: w.id })), [visibleWafers]);
  const rendered = visibleWafers.slice(0, renderLimit);
  const performanceLabel = language === 'ko' ? ({ turbo: '터보', balanced: '균형', fidelity: '정밀' }[performanceMode]) : performanceMode;

  return <section className="spotfire-lab enterprise-lab" id="simulator">
    <div className="lab-hero-shell">
      <div className="lab-orb" />
      <div className="section-heading lab-heading enterprise-heading">
        <div>
          <p className="eyebrow">{c.eyebrow}</p>
          <h2>{c.title}</h2>
          <p className="muted">{c.body}</p>
        </div>
        <div className="lab-mode">
          <span>{mode === 'mock' ? c.browser : c.fastapi}</span>
          <button className="primary compact" onClick={() => void run()} disabled={loading || saving}>{loading ? c.running : c.run}</button>
          <button className="ghost compact" onClick={() => void saveCurrentSession()} disabled={!result || loading || saving}>{saving ? c.saving : c.save}</button>
        </div>
      </div>
    </div>

    <div className="control-ribbon card" id="simulator-filters">
      <label>{c.scenario}<select value={scenario} onChange={(e) => applyScenario(e.target.value)}>{Object.keys(SCENARIOS).map((key) => <option key={key} value={key}>{scenarioText(key, language).title}</option>)}</select></label>
      <label>{c.performance}<select value={performanceMode} onChange={(e) => setPerformanceMode(e.target.value as SimulatorPerformanceMode)}><option value="turbo">{c.turbo}</option><option value="balanced">{c.balanced}</option><option value="fidelity">{c.fidelity}</option></select></label>
      <label>{c.modelPath}<select value={useModel ? 'model' : 'heuristic'} onChange={(e) => setUseModel(e.target.value === 'model')}><option value="model">{c.useModel}</option><option value="heuristic">{c.heuristic}</option></select></label>
      <label>{c.seed}<input value={seed} type="number" onChange={(e) => setSeed(Number(e.target.value || 0))} /></label>
      <p>{activeScenarioText.description}<br /><span className="dim">{c.previewNote}</span></p>
    </div>

    <div className="enterprise-grid">
      <section className="card simulator-controls pro-controls">
        <div className="section-heading"><div><p className="eyebrow">{c.params}</p><h3>{c.lotGenerator}</h3></div></div>
        <Slider label={c.waferCount} value={waferCount} min={24} max={1000} step={8} onChange={setWaferCount} suffix={c.wafersSuffix} />
        <Slider label={c.mapResolution} value={size} min={40} max={144} step={8} onChange={setSize} suffix=" px" />
        <Slider label={c.noise} value={noise} min={0} max={0.18} step={0.005} onChange={setNoise} format={(v) => pct(v, 1)} />
        <Slider label={c.density} value={density} min={0.55} max={2.2} step={0.05} onChange={setDensity} suffix="x" />
        <Slider label={c.mixed} value={mixed} min={0} max={0.55} step={0.01} onChange={setMixed} format={(v) => pct(v, 0)} />
        <div className="scenario-mix pro-mix">{Object.entries(activeScenario.mix).map(([label, weight]) => <span key={label} style={{ '--chip': labelColor(label) } as CSSProperties}>{labelName(label, language)}<b>{Math.round(weight * 100)}%</b></span>)}</div>
      </section>

      <section className="card executive-summary">
        <div className="section-heading"><div><p className="eyebrow">{c.telemetry}</p><h3>{result ? result.session_id : c.noRun}</h3></div><span className="dim">{result ? dt(result.created_at, language) : c.runInit}</span></div>
        <div className="kpi-wall">
          <MiniKpi label={c.wafers} value={String(result?.summary.total_wafers ?? waferCount)} sub={c.batch} />
          <MiniKpi label={c.avgYield} value={pct(result?.summary.avg_yield)} sub={`${c.loss} ${pp(result?.summary.yield_loss_pp)}`} tone="good" />
          <MiniKpi label={c.p95} value={risk(result?.summary.p95_risk)} sub={c.tail} tone="risk" />
          <MiniKpi label={c.runtime} value={ms(result?.summary.simulation_runtime_ms)} sub={performanceLabel} />
          <MiniKpi label={c.agreement} value={pct(result?.summary.model_agreement)} sub={c.truePred} />
          <MiniKpi label={c.highRisk} value={String(result?.summary.high_risk_count ?? 0)} sub={language === 'ko' ? '위험 >= 45' : 'risk >= 45'} tone="risk" />
        </div>
        {result && <div className="export-actions"><button className="ghost compact" onClick={() => exportWafersCsv(result)}>{c.exportWafers}</button><button className="ghost compact" onClick={() => exportRootCausesCsv(result)}>{c.exportRoot}</button><button className="ghost compact" onClick={() => exportSummaryJson(result)}>{c.exportSummary}</button></div>}
        {result?.persistence_note && <p className="footnote persistence-note">{persistenceText(result.persistence_note, language)}</p>}
        <Suspense fallback={<div className="chartbox pro-chart skeleton-card"><p className="muted center">{c.loadingChart}</p></div>}><PatternDistributionChart result={result} language={language} /></Suspense>
      </section>

      <section className="card rootcause-panel">
        <div className="section-heading"><div><p className="eyebrow">{c.rootQueue}</p><h3>{c.candidates}</h3></div></div>
        <div className="root-list">{(result?.summary.root_causes ?? []).length ? result!.summary.root_causes!.map((item) => <article key={item.entity_id}><span>#{item.rank}</span><div><strong>{item.entity_id}</strong><p>{evidenceText(item.evidence, language)}</p><em>{recommendationText(item.recommendation, language)}</em></div><b>{risk(item.score)}</b></article>) : <p className="muted">{c.rootEmpty}</p>}</div>
        <div className="session-list pro-sessions"><p className="eyebrow">{c.savedSessions}</p>{sessionPage.sessions.length ? sessionPage.sessions.map((s) => <button key={s.session_id} onClick={() => void loadSession(s.session_id)}><span>{s.scenario_name ? scenarioText(s.scenario_name, language).title : language === 'ko' ? '시뮬레이션' : 'simulation'}</span><b>{pct(s.avg_yield)}</b><em>{s.persisted_wafer_count ?? 0}/{s.wafer_count} {c.savedCount}</em></button>) : <p className="muted">{c.saveHint}</p>}<div className="pager"><button className="ghost compact" disabled={!sessionPage.hasPrev} onClick={() => void sessionPage.prev()}>{c.prev}</button><span>{sessionPage.total ? `${sessionPage.offset + 1}-${Math.min(sessionPage.offset + sessionPage.sessions.length, sessionPage.total)} / ${sessionPage.total}` : '0 / 0'}</span><button className="ghost compact" disabled={!sessionPage.hasNext} onClick={() => void sessionPage.next()}>{c.next}</button></div></div>
      </section>
    </div>

    <Suspense fallback={<div className="lab-grid charts-row pro-charts-row"><section className="card simulator-chart wide skeleton-card"><p className="eyebrow">{copy[language].panels.loadingCharts}</p><h3>{c.loadingAnalytics}</h3></section></div>}>
      <SimulatorAnalyticsCharts result={result} scatter={scatter} language={language} />
    </Suspense>

    <section className="card confusion-card" id="diagnostics-panel">
      <div className="section-heading"><div><p className="eyebrow">{c.diagnostics}</p><h3>{c.confusion}</h3></div><span className="dim">{result?.summary.confusion.length ?? 0} {c.occupied}</span></div>
      <ConfusionMatrix cells={result?.summary.confusion ?? []} language={language} />
      <ConfusionLegend language={language} />
    </section>

    <section className="card chamber-map-card">
      <div className="section-heading"><div><p className="eyebrow">{c.chamber}</p><h3>{c.chamberTitle}</h3></div><span className="dim">{result?.summary.chamber_risk?.length ?? 0} {c.entities}</span></div>
      <div className="chamber-heatmap">{(result?.summary.chamber_risk ?? []).slice(0, 24).map((row) => <button key={row.entity_id} style={{ '--heat': `${Math.min(1, row.excursion_score / 100)}` } as CSSProperties} onClick={() => setToolFilter(row.tool_id)}><span>{row.entity_id}</span><strong>{risk(row.excursion_score)}</strong><em>{labelName(row.dominant_label, language)} · {row.wafers} {c.wafersSuffix.trim()}</em></button>)}{!result && <p className="muted">{c.chamberEmpty}</p>}</div>
    </section>

    <section className="card trellis-card pro-trellis-card">
      <div className="section-heading"><div><p className="eyebrow">{c.trellis}</p><h3>{c.trellisTitle}</h3></div><span className="dim">{rendered.length} {c.rendered} / {visibleWafers.length} {c.marked} / {result?.wafers.length ?? 0} {c.total}</span></div>
      <div className="filter-strip pro-filter-strip"><select value={labelFilter} onChange={(e) => setLabelFilter(e.target.value)}><option value="">{c.allPredicted}</option>{LABELS.map((label) => <option key={label} value={label}>{labelName(label, language)}</option>)}</select><select value={toolFilter} onChange={(e) => setToolFilter(e.target.value)}><option value="">{c.allTools}</option>{tools.map((tool) => <option key={tool} value={tool}>{tool}</option>)}</select><select value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)}><option value="all">{c.allRisk}</option><option value="critical">{c.criticalOnly}</option><option value="high">{c.highOnly}</option><option value="low">{c.lowOnly}</option></select><select value={renderLimit} onChange={(e) => setRenderLimit(Number(e.target.value))}><option value={120}>{c.render120}</option><option value={240}>{c.render240}</option><option value={400}>{c.render400}</option></select></div>
      <div className="trellis-grid pro-trellis-grid">{rendered.map((wafer) => <button key={wafer.id} className={`wafer-tile pro-wafer-tile ${selected?.id === wafer.id ? 'selected' : ''} ${severityClass(wafer.severity)}`} onClick={() => { setSelectedId(wafer.id); onMatrixSelect?.(wafer.matrix, wafer.predicted_label); }}><WaferCanvas matrix={wafer.matrix} label={wafer.predicted_label} size={86} language={language} /><span className="tile-head"><i aria-hidden="true" />{wafer.id.split('-').slice(-1)[0]}</span><b style={{ color: labelColor(wafer.predicted_label) }}>{labelName(wafer.predicted_label, language)}</b><em>{severityName(wafer.severity, language)} · R{risk(wafer.risk_score)}</em></button>)}{!visibleWafers.length && <p className="empty">{c.trellisEmpty}</p>}</div>
    </section>

    <section className="card selected-wafer pro-selected-wafer">
      <div className="section-heading"><div><p className="eyebrow">{c.detail}</p><h3>{selected ? selected.id : c.noWafer}</h3></div>{selected && <span className={`severity-pill ${severityClass(selected.severity)}`}>{severityName(selected.severity, language)}</span>}</div>
      {selected ? <div className="selected-grid pro-selected-grid"><div className="large-wafer pro-large-wafer"><WaferCanvas matrix={selected.matrix} label={selected.predicted_label} size={420} language={language} /></div><div className="selected-info"><div className="meta-grid pro-meta-grid"><p><span>{c.lot}</span><strong>{selected.lot_id}</strong></p><p><span>{c.tool}</span><strong>{selected.tool_id}-{selected.chamber_id}</strong></p><p><span>{c.step}</span><strong>{processName(selected.process_step, language)}</strong></p><p><span>{c.truePredShort}</span><strong>{labelName(selected.true_label, language)} → {labelName(selected.predicted_label, language)}</strong></p><p><span>{c.yield}</span><strong>{pct(selected.yield_estimate)}</strong></p><p><span>{c.defectDensity}</span><strong>{pct(selected.defect_density)}</strong></p><p><span>{c.edgeCenter}</span><strong>{pct(selected.edge_concentration)} / {pct(selected.center_concentration)}</strong></p><p><span>{c.riskScore}</span><strong>{risk(selected.risk_score)}</strong></p><p><span>{c.cluster}</span><strong>{pct(selected.cluster_intensity)}</strong></p><p><span>{c.radial}</span><strong>{pct(selected.radial_non_uniformity)}</strong></p></div><div className="topk compact-list pro-topk">{selected.top_k.map((item) => <div className="topk-row" key={String(item.label)}><span><i style={{ background: labelColor(item.label) }} />{labelName(item.label, language)}</span><div><b style={{ width: `${item.probability * 100}%`, background: labelColor(item.label) }} /></div><strong>{pct(item.probability)}</strong></div>)}</div><article className="explain pro-explain"><h3>{causeHintText(selected.root_cause_hint, language) || labelInfo(selected.predicted_label, language).title}</h3><p>{labelInfo(selected.predicted_label, language).cause}</p><p><strong>{c.recommended}</strong> {recommendationText(selected.recommended_action, language) || labelInfo(selected.predicted_label, language).action}</p>{selected.secondary_label && <p><strong>{c.mixedPattern}</strong> {labelName(selected.secondary_label, language)}</p>}</article></div></div> : <p className="muted">{c.detailEmpty}</p>}
    </section>
  </section>;
}

function compactResponseForSave(response: SimulatorResponse): SimulatorResponse {
  const maxSavedWafers = 240;
  const matrixSize = 40;
  return {
    ...response,
    params: { ...response.params, persist: true },
    wafers: response.wafers.slice(0, maxSavedWafers).map((wafer) => ({
      ...wafer,
      matrix: compactMatrix(wafer.matrix, matrixSize),
    })),
  };
}

function compactMatrix(matrix: WaferMatrix, target: number): WaferMatrix {
  if (!matrix.length || matrix.length <= target) return matrix;
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  return Array.from({ length: target }, (_, y) => {
    const sy = Math.round((y / Math.max(1, target - 1)) * (rows - 1));
    return Array.from({ length: target }, (_, x) => {
      const sx = Math.round((x / Math.max(1, target - 1)) * (cols - 1));
      return matrix[sy]?.[sx] ?? 0;
    });
  });
}

async function runViaJob(api: ApiClient, payload: SimulatorRequest, language: Language): Promise<SimulatorResponse> {
  if (!api.startSimulatorJob || !api.getSimulatorJob) return api.runSimulator(payload);
  let job = await api.startSimulatorJob(payload);
  for (let i = 0; i < 80; i += 1) {
    if (job.status === 'succeeded') {
      if (job.result) return job.result;
      if (job.session_id && api.getSimulationSession) return api.getSimulationSession(job.session_id);
      throw new Error(language === 'ko' ? '백그라운드 작업이 끝났지만 세션 ID가 없습니다.' : 'Simulator background job completed but did not return a session id.');
    }
    if (job.status === 'failed') throw new Error(job.error ?? (language === 'ko' ? '백그라운드 시뮬레이터 작업이 실패했습니다.' : 'Simulator background job failed.'));
    await new Promise((resolve) => window.setTimeout(resolve, 500));
    job = await api.getSimulatorJob(job.job_id);
  }
  throw new Error(language === 'ko' ? 'UI 폴링 시간 안에 백그라운드 작업이 끝나지 않았습니다.' : 'Simulator background job did not finish before the UI polling timeout.');
}
