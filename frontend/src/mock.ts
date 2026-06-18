import type { ApiClient, DefectLabel, FeatureResponse, HealthResponse, HistoryPage, ModelMetadata, PredictResponse, PredictionHistoryItem, SimulatorJobStatus, SimulatorRequest, SimulatorResponse, SimulatorSessionPage, SimulatorWafer, StatsSummary, TopKPrediction, WaferMatrix } from './types';
import { LABELS, createPattern, matrixStats } from './wafer';

const KEY = 'wafervision.week3.history';
let latestPreview: WaferMatrix | null = createPattern('Edge-Ring');

const FEATURE_SCHEMA = [
  ...['top', 'right', 'bottom', 'left', 'r1c1', 'r1c2', 'r1c3', 'r2c1', 'r2c2', 'r2c3', 'r3c1', 'r3c2', 'r3c3'].map((x) => `region_${x}_density`),
  ...Array.from({ length: 20 }, (_, i) => `radon_mean_${String(i).padStart(2, '0')}`),
  ...Array.from({ length: 20 }, (_, i) => `radon_std_${String(i).padStart(2, '0')}`),
  'geom_area_norm', 'geom_perimeter_norm', 'geom_major_axis_norm', 'geom_minor_axis_norm', 'geom_eccentricity', 'geom_solidity',
];

function mockFeatures(matrix: WaferMatrix | null, filename = 'mock-wafer.csv'): FeatureResponse {
  const s = matrixStats(matrix);
  const region = Array.from({ length: 13 }, (_, i) => Number(((s.defectRatio * 100) * (0.62 + (i % 5) * 0.11)).toFixed(4)));
  const radonMean = Array.from({ length: 20 }, (_, i) => Number((s.defectRatio * (1 + Math.sin(i / 3) * 0.25)).toFixed(6)));
  const radonStd = Array.from({ length: 20 }, (_, i) => Number((s.scratchScore * (1 + Math.cos(i / 4) * 0.2)).toFixed(6)));
  const geometry = [s.defectRatio, s.edgeRatio, s.centerRatio, Math.max(0.01, s.scratchScore), Math.min(0.99, s.scratchScore * 4), s.defectRatio > 0 ? 0.78 : 0];
  const vector = [...region, ...radonMean, ...radonStd, ...geometry];
  return {
    feature_dim: 59,
    feature_schema: FEATURE_SCHEMA,
    vector,
    named_vector: Object.fromEntries(FEATURE_SCHEMA.map((name, i) => [name, vector[i] ?? 0])),
    groups: [
      { name: 'region_density', values: region, labels: FEATURE_SCHEMA.slice(0, 13) },
      { name: 'radon_mean', values: radonMean, labels: FEATURE_SCHEMA.slice(13, 33) },
      { name: 'radon_std', values: radonStd, labels: FEATURE_SCHEMA.slice(33, 53) },
      { name: 'geometry', values: geometry, labels: FEATURE_SCHEMA.slice(53) },
    ],
    input: { filename, content_type: 'browser/mock', input_kind: 'mock-feature-array', rows: matrix?.length ?? 64, cols: matrix?.[0]?.length ?? 64, min_value: 0, max_value: 2 },
  };
}

export function setMockPreview(matrix: WaferMatrix | null) { latestPreview = matrix; }

function model(): ModelMetadata {
  return {
    loaded: true,
    model_version: 'kaggle-59d-ovo-linear-svm-demo',
    checkpoint_path: 'mock://browser-demo',
    input_size: null,
    class_names: LABELS,
    device: 'browser-mock',
    validation_macro_f1: 0.790,
    model_kind: 'kaggle_svm_ovo',
    feature_dim: 59,
    feature_schema: FEATURE_SCHEMA,
  };
}

function seed(): PredictionHistoryItem[] {
  const now = Date.now();
  const labels = ['Edge-Ring', 'Scratch', 'Center', 'Random', 'Loc', 'None', 'Donut', 'Edge-Loc'];
  return labels.flatMap((label, i) => Array.from({ length: i % 3 === 0 ? 3 : 2 }, (_, n) => ({
    id: i * 10 + n + 1,
    created_at: new Date(now - (i * 2 + n) * 36 * 60 * 1000).toISOString(),
    filename: `${label.toLowerCase()}-lot-${i + 1}-${n + 1}.csv`,
    input_kind: n % 2 === 0 ? 'upload' : 'inline-array',
    predicted_label: label,
    confidence: Math.min(0.98, 0.71 + i * 0.018 + n * 0.021),
    inference_ms: 16 + i * 3 + n,
    rows: 64,
    cols: 64,
    model_version: model().model_version,
  })));
}

function load(): PredictionHistoryItem[] {
  const raw = localStorage.getItem(KEY);
  if (!raw) return seed();
  try { return JSON.parse(raw) as PredictionHistoryItem[]; } catch { return seed(); }
}
function save(items: PredictionHistoryItem[]) { localStorage.setItem(KEY, JSON.stringify(items.slice(0, 200))); }
function wait(ms?: number) { return new Promise((resolve) => window.setTimeout(resolve, ms ?? (320 + Math.random() * 420))); }
function yieldToBrowser() {
  return new Promise<void>((resolve) => {
    // Let React paint the "Running lot…" state before the next generator chunk.
    // requestAnimationFrame gives the browser a visual frame; setTimeout keeps
    // behavior safe in older/headless environments.
    if (typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(() => window.setTimeout(resolve, 0));
    } else {
      window.setTimeout(resolve, 0);
    }
  });
}


function infer(matrix: WaferMatrix | null, filename = ''): DefectLabel {
  const lower = filename.toLowerCase();
  const named = LABELS.find((label) => lower.includes(String(label).toLowerCase()) || lower.includes(String(label).toLowerCase().replace('-', '_')));
  if (named) return named;
  const s = matrixStats(matrix);
  if (s.defectRatio > 0.58) return 'Near-full';
  if (s.scratchScore > 0.12) return 'Scratch';
  if (s.edgeRatio > 0.35) return 'Edge-Ring';
  if (s.centerRatio > 0.3) return 'Center';
  if (s.defectRatio < 0.018) return 'None';
  if (s.defectRatio > 0.09) return 'Loc';
  return 'Random';
}

function topK(label: DefectLabel, matrix: WaferMatrix | null): TopKPrediction[] {
  const s = matrixStats(matrix);
  const first = Math.max(0.68, Math.min(0.97, 0.72 + s.defectRatio * 0.55));
  const raw = [first, 0.12, 0.07, 0.04, 0.02];
  const total = raw.reduce((a, b) => a + b, 0);
  return [label, ...LABELS.filter((item) => item !== label).slice(0, 4)].map((item, i) => ({ label: item, probability: raw[i] / total }));
}

function severity(risk: number) {
  if (risk >= 72) return 'Critical';
  if (risk >= 45) return 'Warning';
  if (risk >= 24) return 'Monitor';
  return 'Normal';
}

function causeFor(label: DefectLabel, tool: string, chamber: string) {
  const map: Record<string, [string, string]> = {
    Center: ['Center zone process bias', 'Check etch/deposition center uniformity, chuck contact, and center gas flow.'],
    Donut: ['Radial ring non-uniformity', 'Review deposition/CVD radial profile, temperature ring, and recipe transitions.'],
    'Edge-Loc': ['Localized edge excursion', 'Inspect edge exclusion, bevel clean, clamp ring, and chamber edge effects.'],
    'Edge-Ring': ['Edge ring excursion', 'Check etch edge uniformity, focus ring wear, and edge gas flow calibration.'],
    Loc: ['Localized contamination or lithography issue', 'Review reticle/track contamination, local particles, and lot handling records.'],
    'Near-full': ['Global catastrophic excursion', 'Hold lot, verify metrology, and check upstream process/tool recipe integrity.'],
    Random: ['Random particle signature', 'Compare particle monitor, CMP/clean logs, and recent maintenance events.'],
    Scratch: ['Mechanical handling scratch', 'Inspect robot arm, FOUP/slot, clean/CMP handling path, and wafer transfer logs.'],
    None: ['No dominant spatial pattern', 'Continue monitoring; compare with control lot and baseline SPC limits.'],
  };
  const [title, action] = map[String(label)] ?? map.Random;
  return [`${title} @ ${tool}-${chamber}`, action] as const;
}

function downsample(matrix: WaferMatrix, target = 56): WaferMatrix {
  if (matrix.length <= target && (matrix[0]?.length ?? 0) <= target) return matrix;
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

function summarize(items: PredictionHistoryItem[]): StatsSummary {
  const counts = new Map<string, number>();
  for (const item of items) counts.set(item.predicted_label, (counts.get(item.predicted_label) ?? 0) + 1);
  return {
    total_predictions: items.length,
    average_confidence: items.length ? items.reduce((sum, item) => sum + item.confidence, 0) / items.length : null,
    label_counts: [...counts.entries()].map(([label, count]) => ({ label, count })).sort((a, b) => b.count - a.count),
    latest: items.slice(0, 10),
  };
}

function record(label: DefectLabel, matrix: WaferMatrix | null, filename: string, kind: string): PredictResponse {
  const items = load();
  const id = Math.max(0, ...items.map((item) => item.id)) + 1;
  const created = new Date().toISOString();
  const ks = topK(label, matrix);
  const inferenceMs = 14 + Math.random() * 18;
  const item: PredictionHistoryItem = {
    id,
    created_at: created,
    filename,
    input_kind: kind,
    predicted_label: label,
    confidence: ks[0].probability,
    inference_ms: inferenceMs,
    rows: matrix?.length ?? 64,
    cols: matrix?.[0]?.length ?? 64,
    model_version: model().model_version,
  };
  save([item, ...items]);
  return {
    id,
    label,
    confidence: ks[0].probability,
    top_k: ks,
    inference_ms: inferenceMs,
    created_at: created,
    input: { filename, content_type: kind === 'upload' ? 'browser/file' : 'application/json', input_kind: kind, rows: item.rows, cols: item.cols, min_value: 0, max_value: 2 },
    model: model(),
  };
}


const SIM_KEY = 'wafervision.spotfire.sessions';

function seededRandom(seed = 1) {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function weightedPick(weights: Record<string, number>, rnd: () => number): DefectLabel {
  const entries = LABELS.map((label) => [label, Math.max(0, Number(weights[label] ?? 0))] as const).filter(([, value]) => value > 0);
  const source = entries.length ? entries : LABELS.map((label) => [label, 1] as const);
  const total = source.reduce((sum, [, value]) => sum + value, 0);
  let cursor = rnd() * total;
  for (const [label, value] of source) {
    cursor -= value;
    if (cursor <= 0) return label;
  }
  return source[source.length - 1][0];
}

function noisyPattern(label: DefectLabel, size: number, noise: number, density: number, rnd: () => number): WaferMatrix {
  const base = createPattern(label, size);
  const c = (size - 1) / 2;
  const radius = size * 0.46;
  return base.map((row, y) => row.map((value, x) => {
    const inside = Math.hypot(x - c, y - c) <= radius;
    if (!inside) return 0;
    if (value >= 2 && rnd() > Math.min(0.45, Math.max(0, density - 0.55) * 0.18)) return 2;
    const p = label === 'None' ? noise * 0.18 : noise * density;
    return rnd() < p ? 2 : Math.max(1, value);
  }));
}

function simulateOne(payload: SimulatorRequest, index: number, rnd: () => number, sessionSeed: string): SimulatorWafer {
  const mix = payload.pattern_mix ?? { 'Edge-Ring': 0.2, Loc: 0.16, Random: 0.16, 'Edge-Loc': 0.13, Center: 0.1, Scratch: 0.1, Donut: 0.07, 'Near-full': 0.04, None: 0.04 };
  const label = weightedPick(mix, rnd);
  const matrixSeed = Math.floor(rnd() * 2 ** 32);
  const matrixRnd = seededRandom(matrixSeed);
  const matrix = noisyPattern(label, payload.size || 64, payload.noise_level, payload.defect_density_scale, matrixRnd);
  const predicted = infer(matrix, String(label));
  const ks = topK(predicted, matrix);
  const s = matrixStats(matrix);
  const lotCount = payload.lot_count || 3;
  const lotId = `LOT-${sessionSeed}-${(index % lotCount) + 1}`;
  const tools = ['ETCH-01', 'ETCH-02', 'LITHO-03', 'CMP-04', 'CVD-05', 'CLEAN-06'];
  const steps = ['Lithography', 'Etch', 'Deposition', 'CMP', 'Clean', 'Metrology'];
  const tool = label === 'Scratch' ? 'CLEAN-06' : label === 'Edge-Ring' || label === 'Edge-Loc' ? tools[index % 2] : tools[Math.floor(rnd() * tools.length)];
  const chamber = ['A', 'B', 'C', 'D'][index % 4];
  const risk = Math.min(100, (s.defectRatio * 135 + Math.max(s.edgeRatio, s.centerRatio) * 25 + s.scratchScore * 58 + Math.abs(s.edgeRatio - s.centerRatio) * 55 + (1 - ks[0].probability) * 12));
  const [rootCause, action] = causeFor(predicted, tool, chamber);
  const returnSize = payload.performance_mode === 'turbo' ? 40 : payload.return_matrix_size ?? 56;
  return {
    id: `${lotId}-W${String(index + 1).padStart(3, '0')}`,
    lot_id: lotId,
    wafer_index: index + 1,
    tool_id: tool,
    chamber_id: chamber,
    process_step: label === 'Scratch' ? 'Clean' : label === 'Edge-Ring' ? 'Etch' : steps[Math.floor(rnd() * steps.length)],
    true_label: label,
    predicted_label: predicted,
    confidence: ks[0].probability,
    top_k: ks,
    defect_density: s.defectRatio,
    yield_estimate: Math.max(0, Math.min(1, 1 - s.defectRatio * 0.86)),
    edge_concentration: s.edgeRatio,
    center_concentration: s.centerRatio,
    scratch_score: s.scratchScore,
    radial_non_uniformity: Math.abs(s.edgeRatio - s.centerRatio),
    risk_score: risk,
    edge_center_delta: s.edgeRatio - s.centerRatio,
    cluster_intensity: Math.min(1, s.defectRatio * 3.5),
    severity: severity(risk),
    root_cause_hint: rootCause,
    recommended_action: action,
    secondary_label: null,
    matrix_seed: matrixSeed,
    matrix: downsample(matrix, returnSize),
  };
}

function summarizeSimulation(wafers: SimulatorWafer[], runtimeMs = 0) {
  const countMap = new Map<string, number>();
  const trueMap = new Map<string, number>();
  const toolMap = new Map<string, SimulatorWafer[]>();
  const chamberMap = new Map<string, SimulatorWafer[]>();
  const confusion = new Map<string, number>();
  wafers.forEach((w) => {
    countMap.set(String(w.predicted_label), (countMap.get(String(w.predicted_label)) ?? 0) + 1);
    trueMap.set(String(w.true_label), (trueMap.get(String(w.true_label)) ?? 0) + 1);
    toolMap.set(w.tool_id, [...(toolMap.get(w.tool_id) ?? []), w]);
    chamberMap.set(`${w.tool_id}-${w.chamber_id}`, [...(chamberMap.get(`${w.tool_id}-${w.chamber_id}`) ?? []), w]);
    const key = `${w.true_label}|||${w.predicted_label}`;
    confusion.set(key, (confusion.get(key) ?? 0) + 1);
  });
  const n = Math.max(1, wafers.length);
  const risks = wafers.map((w) => w.risk_score).sort((a, b) => a - b);
  const avgYield = wafers.reduce((sum, w) => sum + w.yield_estimate, 0) / n;
  const chamberRisk = [...chamberMap.entries()].map(([entity_id, rows]) => {
    const dominant = [...rows.reduce((m, w) => m.set(String(w.predicted_label), (m.get(String(w.predicted_label)) ?? 0) + 1), new Map<string, number>()).entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'Random';
    const avgRisk = rows.reduce((sum, w) => sum + w.risk_score, 0) / Math.max(1, rows.length);
    const highShare = rows.filter((w) => w.risk_score >= 45).length / Math.max(1, rows.length);
    const [tool_id, chamber_id] = entity_id.split('-');
    return {
      tool_id: `${tool_id}-${chamber_id}`,
      chamber_id: entity_id.split('-').slice(-1)[0],
      entity_id,
      wafers: rows.length,
      avg_risk: avgRisk,
      avg_yield: rows.reduce((sum, w) => sum + w.yield_estimate, 0) / Math.max(1, rows.length),
      dominant_label: dominant,
      excursion_score: Math.min(100, avgRisk * 0.72 + highShare * 32 + Math.log1p(rows.length) * 2.2),
    };
  }).sort((a, b) => b.excursion_score - a.excursion_score);
  return {
    total_wafers: wafers.length,
    avg_yield: avgYield,
    avg_confidence: wafers.reduce((sum, w) => sum + w.confidence, 0) / n,
    avg_defect_density: wafers.reduce((sum, w) => sum + w.defect_density, 0) / n,
    high_risk_count: wafers.filter((w) => w.risk_score >= 45).length,
    p95_risk: risks.length ? risks[Math.min(risks.length - 1, Math.floor(risks.length * 0.95))] : 0,
    yield_loss_pp: (1 - avgYield) * 100,
    model_agreement: wafers.filter((w) => w.true_label === w.predicted_label).length / n,
    simulation_runtime_ms: runtimeMs,
    label_counts: [...countMap.entries()].map(([label, count]) => ({ label, count })).sort((a, b) => b.count - a.count),
    true_label_counts: [...trueMap.entries()].map(([label, count]) => ({ label, count })).sort((a, b) => b.count - a.count),
    tool_risk: [...toolMap.entries()].map(([tool_id, rows]) => ({
      tool_id,
      wafers: rows.length,
      avg_risk: rows.reduce((sum, w) => sum + w.risk_score, 0) / Math.max(1, rows.length),
      avg_yield: rows.reduce((sum, w) => sum + w.yield_estimate, 0) / Math.max(1, rows.length),
      top_pattern: [...rows.reduce((m, w) => m.set(String(w.predicted_label), (m.get(String(w.predicted_label)) ?? 0) + 1), new Map<string, number>()).entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'Random',
    })).sort((a, b) => b.avg_risk - a.avg_risk),
    chamber_risk: chamberRisk.slice(0, 24),
    root_causes: chamberRisk.slice(0, 6).map((row, i) => {
      const rows = chamberMap.get(row.entity_id) ?? [];
      const sample = rows[0];
      return { rank: i + 1, entity_type: 'tool_chamber', entity_id: row.entity_id, score: row.excursion_score, evidence: `${row.wafers} wafers, avg risk ${row.avg_risk.toFixed(1)}, dominant pattern ${row.dominant_label}.`, recommendation: sample?.recommended_action ?? 'Compare lot, tool, chamber, and process step.' };
    }),
    confusion: [...confusion.entries()].map(([key, count]) => { const [true_label, predicted_label] = key.split('|||'); return { true_label, predicted_label, count }; }),
    trend: wafers.map((w, i) => ({ index: i + 1, lot_id: w.lot_id, wafer_id: w.id, yield_estimate: w.yield_estimate, defect_density: w.defect_density, risk_score: w.risk_score, predicted_label: w.predicted_label })),
  };
}


function loadSessions(): SimulatorResponse[] {
  const raw = localStorage.getItem(SIM_KEY);
  if (!raw) return [];
  try { return JSON.parse(raw) as SimulatorResponse[]; } catch { return []; }
}

function compactSession(response: SimulatorResponse): SimulatorResponse {
  const maxSavedWafers = 240;
  const matrixSize = 36;
  const wafers = response.wafers.slice(0, maxSavedWafers).map((wafer) => ({
    ...wafer,
    matrix: downsample(wafer.matrix, matrixSize),
  }));
  return {
    ...response,
    wafers,
    persisted_wafer_count: wafers.length,
    persistence_note: `Loaded compact browser session: ${wafers.length} of ${response.summary.total_wafers} wafers with ${matrixSize}px matrices. Full aggregate summary is preserved.`,
  };
}

function saveSession(response: SimulatorResponse): SimulatorResponse {
  const compact = compactSession(response);
  const sessions = [compact, ...loadSessions().filter((s) => s.session_id !== response.session_id)].slice(0, 12);
  localStorage.setItem(SIM_KEY, JSON.stringify(sessions));
  return compact;
}


async function runMockSimulator(payload: SimulatorRequest): Promise<SimulatorResponse> {
  const started = performance.now();
  const rnd = seededRandom(payload.seed ?? 20260617);
  const suffix = String(Math.floor(rnd() * 9000) + 1000);
  const sessionId = `sim-browser-${Date.now().toString(36)}-${suffix}`;
  const wafers: SimulatorWafer[] = [];
  const chunkSize = payload.performance_mode === 'turbo' ? 96 : payload.performance_mode === 'fidelity' ? 18 : 36;
  let lastYield = performance.now();
  for (let index = 0; index < payload.wafer_count; index += 1) {
    wafers.push(simulateOne(payload, index, rnd, suffix));
    const shouldYield = (index + 1) % chunkSize === 0 || performance.now() - lastYield > 12;
    if (shouldYield && index + 1 < payload.wafer_count) {
      await yieldToBrowser();
      lastYield = performance.now();
    }
  }
  const shouldPersist = payload.persist === true;
  const response: SimulatorResponse = {
    session_id: sessionId,
    created_at: new Date().toISOString(),
    params: payload,
    wafers,
    summary: summarizeSimulation(wafers, performance.now() - started),
    model: model(),
    persisted_wafer_count: shouldPersist ? Math.min(wafers.length, 240) : 0,
    persistence_note: shouldPersist
      ? 'Browser demo stored this compact session in localStorage.'
      : 'Preview run only: session was not persisted. Use Save current session to store it.',
  };
  if (shouldPersist) {
    const saved = saveSession(response);
    return { ...response, persisted_wafer_count: saved.persisted_wafer_count, persistence_note: 'Saved compact browser snapshot; current preview keeps full returned wafers.' };
  }
  return response;
}

export const mockClient: ApiClient = {
  async getHealth(): Promise<HealthResponse> { return { status: 'ok', app: 'WaferVision Mock API', environment: 'browser-demo', model_loaded: true, database: 'localStorage://history' }; },
  async getModel() { return model(); },
  async getStats() { return summarize(load()); },
  async listPredictions({ limit = 50, offset = 0, label } = {}): Promise<HistoryPage> {
    const items = load().filter((item) => !label || item.predicted_label === label);
    return { total: items.length, limit, offset, items: items.slice(offset, offset + limit) };
  },
  async predictFile(file: File) { await wait(); return record(infer(latestPreview, file.name), latestPreview, file.name, 'upload'); },
  async predictArray(matrix: WaferMatrix, filename = 'inline-wafer-map.csv') { await wait(); return record(infer(matrix, filename), matrix, filename, 'inline-array'); },
  async extractFeaturesFile(file: File) { await wait(); return mockFeatures(latestPreview, file.name); },
  async extractFeaturesArray(matrix: WaferMatrix, filename = 'inline-wafer-map.csv') { await wait(); return mockFeatures(matrix, filename); },
  async runSimulator(payload: SimulatorRequest) { await wait(40); return runMockSimulator(payload); },
  async saveSimulationSession(response: SimulatorResponse) {
    await yieldToBrowser();
    const saved = saveSession(response);
    return { ...response, persisted_wafer_count: saved.persisted_wafer_count, persistence_note: saved.persistence_note };
  },
  async listSimulationSessions({ limit = 20, offset = 0 } = {}): Promise<SimulatorSessionPage> {
    const sessions = loadSessions();
    return {
      total: sessions.length,
      limit,
      offset,
      items: sessions.slice(offset, offset + limit).map((s) => ({
        session_id: s.session_id,
        created_at: s.created_at,
        scenario_name: s.params.scenario_name ?? null,
        wafer_count: s.summary.total_wafers,
        persisted_wafer_count: s.persisted_wafer_count ?? s.wafers.length,
        avg_yield: s.summary.avg_yield,
        high_risk_count: s.summary.high_risk_count,
      })),
    };
  },
  async getSimulationSession(sessionId: string) {
    const found = loadSessions().find((s) => s.session_id === sessionId);
    if (!found) throw new Error('Simulation session not found.');
    return found;
  },
  async startSimulatorJob(payload: SimulatorRequest): Promise<SimulatorJobStatus> {
    await wait(120);
    const result = await runMockSimulator(payload);
    return {
      job_id: `job-browser-${Date.now().toString(36)}`,
      status: 'succeeded',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      progress: 1,
      session_id: result.session_id,
      result,
    };
  },
  async getSimulatorJob(jobId: string): Promise<SimulatorJobStatus> {
    await wait(80);
    return { job_id: jobId, status: 'failed', created_at: new Date().toISOString(), updated_at: new Date().toISOString(), progress: 1, error: 'Browser demo jobs are completed immediately from startSimulatorJob().' };
  },
};
