export type DefectLabel =
  | 'Center'
  | 'Donut'
  | 'Edge-Loc'
  | 'Edge-Ring'
  | 'Loc'
  | 'Near-full'
  | 'Random'
  | 'Scratch'
  | 'None'
  | string;

export type WaferMatrix = number[][];

export interface TopKPrediction {
  label: DefectLabel;
  probability: number;
}

export interface InputMetadata {
  filename: string | null;
  content_type: string | null;
  input_kind: string;
  rows: number | null;
  cols: number | null;
  min_value: number | null;
  max_value: number | null;
}

export interface ModelMetadata {
  loaded: boolean;
  model_version: string;
  checkpoint_path: string;
  input_size: number | null;
  class_names: DefectLabel[];
  device: string | null;
  validation_macro_f1: number | null;
  model_kind?: string;
  feature_dim?: number | null;
  feature_schema?: string[];
  load_error?: string | null;
}


export interface FeatureGroup {
  name: string;
  values: number[];
  labels: string[];
}

export interface FeatureResponse {
  feature_dim: number;
  feature_schema: string[];
  groups: FeatureGroup[];
  vector: number[];
  named_vector: Record<string, number>;
  input: InputMetadata;
}

export interface PredictResponse {
  id: number;
  label: DefectLabel;
  confidence: number;
  top_k: TopKPrediction[];
  inference_ms: number;
  created_at: string;
  input: InputMetadata;
  model: ModelMetadata;
}

export interface PredictionHistoryItem {
  id: number;
  created_at: string;
  filename: string | null;
  input_kind: string;
  predicted_label: DefectLabel;
  confidence: number;
  inference_ms: number;
  rows: number | null;
  cols: number | null;
  model_version: string;
}

export interface HistoryPage {
  total: number;
  limit: number;
  offset: number;
  items: PredictionHistoryItem[];
}

export interface StatsSummary {
  total_predictions: number;
  average_confidence: number | null;
  label_counts: Array<{ label: DefectLabel; count: number }>;
  latest: PredictionHistoryItem[];
}

export interface HealthResponse {
  status: string;
  app: string;
  environment: string;
  model_loaded: boolean;
  database: string;
}

export interface ApiClient {
  getHealth(): Promise<HealthResponse>;
  getModel(): Promise<ModelMetadata>;
  getStats(): Promise<StatsSummary>;
  listPredictions(params?: { limit?: number; offset?: number; label?: string }): Promise<HistoryPage>;
  predictFile(file: File, note?: string): Promise<PredictResponse>;
  predictArray(matrix: WaferMatrix, filename?: string, note?: string): Promise<PredictResponse>;
  extractFeaturesFile(file: File): Promise<FeatureResponse>;
  extractFeaturesArray(matrix: WaferMatrix, filename?: string): Promise<FeatureResponse>;
  runSimulator(payload: SimulatorRequest): Promise<SimulatorResponse>;
  saveSimulationSession(response: SimulatorResponse): Promise<SimulatorResponse>;
  listSimulationSessions(params?: { limit?: number; offset?: number }): Promise<SimulatorSessionPage>;
  getSimulationSession(sessionId: string): Promise<SimulatorResponse>;
  startSimulatorJob?(payload: SimulatorRequest): Promise<SimulatorJobStatus>;
  getSimulatorJob?(jobId: string): Promise<SimulatorJobStatus>;
}

export type SimulatorPerformanceMode = 'turbo' | 'balanced' | 'fidelity';

export interface SimulatorRequest {
  wafer_count: number;
  size: number;
  seed?: number | null;
  lot_count?: number;
  noise_level: number;
  defect_density_scale: number;
  mixed_pattern_rate: number;
  pattern_mix?: Record<string, number> | null;
  persist?: boolean;
  scenario_name?: string | null;
  use_model?: boolean;
  performance_mode?: SimulatorPerformanceMode;
  return_matrix_size?: number | null;
  downsample_method?: 'nearest' | 'area';
  model_batch_size?: number | null;
  risk_weights?: Record<string, number> | null;
  severity_thresholds?: Record<string, number> | null;
}

export interface SimulatorWafer {
  id: string;
  lot_id: string;
  wafer_index: number;
  tool_id: string;
  chamber_id: string;
  process_step: string;
  true_label: DefectLabel;
  predicted_label: DefectLabel;
  confidence: number;
  top_k: TopKPrediction[];
  defect_density: number;
  yield_estimate: number;
  edge_concentration: number;
  center_concentration: number;
  scratch_score: number;
  radial_non_uniformity: number;
  risk_score: number;
  edge_center_delta?: number;
  cluster_intensity?: number;
  severity?: 'Normal' | 'Monitor' | 'Warning' | 'Critical' | string;
  root_cause_hint?: string | null;
  recommended_action?: string | null;
  secondary_label?: DefectLabel | null;
  matrix_seed?: number | null;
  matrix: WaferMatrix;
}

export interface SimulatorLabelCount { label: DefectLabel; count: number }
export interface SimulatorToolMetric { tool_id: string; wafers: number; avg_risk: number; avg_yield: number; top_pattern: DefectLabel }
export interface SimulatorChamberMetric { tool_id: string; chamber_id: string; entity_id: string; wafers: number; avg_risk: number; avg_yield: number; dominant_label: DefectLabel; excursion_score: number }
export interface SimulatorRootCause { rank: number; entity_type: string; entity_id: string; score: number; evidence: string; recommendation: string }
export interface SimulatorConfusionCell { true_label: DefectLabel; predicted_label: DefectLabel; count: number }
export interface SimulatorTrendPoint { index: number; lot_id: string; wafer_id: string; yield_estimate: number; defect_density: number; risk_score: number; predicted_label: DefectLabel }

export interface SimulatorSummary {
  total_wafers: number;
  avg_yield: number;
  avg_confidence: number;
  avg_defect_density: number;
  high_risk_count: number;
  p95_risk?: number;
  yield_loss_pp?: number;
  model_agreement?: number | null;
  simulation_runtime_ms?: number | null;
  label_counts: SimulatorLabelCount[];
  true_label_counts: SimulatorLabelCount[];
  tool_risk: SimulatorToolMetric[];
  chamber_risk?: SimulatorChamberMetric[];
  root_causes?: SimulatorRootCause[];
  confusion: SimulatorConfusionCell[];
  trend: SimulatorTrendPoint[];
}

export interface SimulatorResponse {
  session_id: string;
  created_at: string;
  params: SimulatorRequest;
  wafers: SimulatorWafer[];
  summary: SimulatorSummary;
  model: ModelMetadata;
  persisted_wafer_count?: number | null;
  persistence_note?: string | null;
}

export interface SimulatorSessionListItem {
  session_id: string;
  created_at: string;
  scenario_name: string | null;
  wafer_count: number;
  persisted_wafer_count?: number | null;
  avg_yield: number | null;
  high_risk_count: number | null;
}

export interface SimulatorSessionPage {
  total: number;
  limit: number;
  offset: number;
  items: SimulatorSessionListItem[];
}

export interface SimulatorJobStatus {
  job_id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | string;
  created_at: string;
  updated_at: string;
  progress: number;
  session_id?: string | null;
  error?: string | null;
  result?: SimulatorResponse | null;
  backend?: string | null;
  queue_name?: string | null;
  external_job_id?: string | null;
  result_available?: boolean | null;
}
