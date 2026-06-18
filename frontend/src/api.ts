import type { ApiClient, FeatureResponse, HealthResponse, HistoryPage, ModelMetadata, PredictResponse, SimulatorJobStatus, SimulatorRequest, SimulatorResponse, SimulatorSessionPage, StatsSummary, WaferMatrix } from './types';

function truthy(value: unknown): boolean {
  return ['1', 'true', 'yes', 'on'].includes(String(value ?? '').toLowerCase());
}

function normalizeApiBase(raw?: string): string {
  const value = (raw || 'http://localhost:8000').replace(/\/+$/, '');
  return value.endsWith('/api/v1') ? value : `${value}/api/v1`;
}

export const config = {
  apiBaseUrl: normalizeApiBase(import.meta.env.VITE_API_BASE_URL),
  useMocks: truthy(import.meta.env.VITE_USE_MOCKS ?? import.meta.env.VITE_DEMO_MODE ?? 'true'),
  timeoutMs: Number(import.meta.env.VITE_API_TIMEOUT_MS || 15000),
  apiKey: String(import.meta.env.VITE_API_KEY || ''),
};

function withAuthHeaders(init: RequestInit = {}): RequestInit {
  if (!config.apiKey) return init;
  const headers = new Headers(init.headers);
  headers.set('x-api-key', config.apiKey);
  return { ...init, headers };
}

async function fetchTimeout(url: string, init: RequestInit = {}) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), config.timeoutMs);
  try {
    return await fetch(url, { ...withAuthHeaders(init), signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}


function downsampleForSave(matrix: WaferMatrix, target = 40): WaferMatrix {
  if (!matrix.length || matrix.length <= target && (matrix[0]?.length ?? 0) <= target) return matrix;
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

function compactSimulationForSave(response: SimulatorResponse): SimulatorResponse {
  const maxWafers = 240;
  const matrixSize = 40;
  return {
    ...response,
    params: { ...response.params, persist: true },
    wafers: response.wafers.slice(0, maxWafers).map((wafer) => ({ ...wafer, matrix: downsampleForSave(wafer.matrix, matrixSize) })),
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetchTimeout(`${config.apiBaseUrl}${path}`, init);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const payload = await res.json();
      message = typeof payload.detail === 'string' ? payload.detail : JSON.stringify(payload.detail ?? payload);
    } catch {
      message = await res.text();
    }
    throw new Error(message || 'API request failed.');
  }
  return res.json() as Promise<T>;
}

export const realClient: ApiClient = {
  getHealth: () => request<HealthResponse>('/health'),
  getModel: () => request<ModelMetadata>('/model'),
  getStats: () => request<StatsSummary>('/stats/summary'),
  listPredictions: ({ limit = 50, offset = 0, label } = {}) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (label) params.set('label', label);
    return request<HistoryPage>(`/predictions?${params.toString()}`);
  },
  predictFile: (file: File, note?: string) => {
    const form = new FormData();
    form.append('file', file);
    if (note) form.append('note', note);
    return request<PredictResponse>('/predict', { method: 'POST', body: form });
  },
  predictArray: (matrix: WaferMatrix, filename = 'inline-wafer-map.csv', note?: string) =>
    request<PredictResponse>('/predict/array', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wafer_map: matrix, filename, note }),
    }),
  extractFeaturesFile: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return request<FeatureResponse>('/features', { method: 'POST', body: form });
  },
  extractFeaturesArray: (matrix: WaferMatrix, filename = 'inline-wafer-map.csv') =>
    request<FeatureResponse>('/features/array', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wafer_map: matrix, filename }),
    }),
  runSimulator: (payload: SimulatorRequest) =>
    request<SimulatorResponse>('/simulator/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  listSimulationSessions: ({ limit = 20, offset = 0 } = {}) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    return request<SimulatorSessionPage>(`/simulator/sessions?${params.toString()}`);
  },
  getSimulationSession: (sessionId: string) => request<SimulatorResponse>(`/simulator/sessions/${encodeURIComponent(sessionId)}`),
  saveSimulationSession: (response: SimulatorResponse) =>
    request<SimulatorResponse>('/simulator/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(compactSimulationForSave(response)),
    }),
  startSimulatorJob: (payload: SimulatorRequest) =>
    request<SimulatorJobStatus>('/simulator/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  getSimulatorJob: (jobId: string) => request<SimulatorJobStatus>(`/simulator/jobs/${encodeURIComponent(jobId)}`),
};
