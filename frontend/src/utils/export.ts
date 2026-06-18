import type { SimulatorResponse, SimulatorRootCause, SimulatorWafer } from '../types';

function csvEscape(value: unknown): string {
  if (value === null || value === undefined) return '';
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function downloadText(filename: string, text: string, mime = 'text/csv;charset=utf-8') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function exportWafersCsv(result: SimulatorResponse) {
  const headers: Array<keyof SimulatorWafer | 'tool_chamber'> = [
    'id', 'lot_id', 'wafer_index', 'tool_id', 'chamber_id', 'tool_chamber', 'process_step',
    'true_label', 'predicted_label', 'confidence', 'yield_estimate', 'defect_density',
    'edge_concentration', 'center_concentration', 'scratch_score', 'radial_non_uniformity',
    'cluster_intensity', 'risk_score', 'severity', 'root_cause_hint', 'recommended_action', 'secondary_label',
  ];
  const rows = result.wafers.map((w) => headers.map((h) => {
    if (h === 'tool_chamber') return csvEscape(`${w.tool_id}-${w.chamber_id}`);
    return csvEscape(w[h as keyof SimulatorWafer]);
  }).join(','));
  downloadText(`${result.session_id}-wafers.csv`, [headers.join(','), ...rows].join('\n'));
}

export function exportRootCausesCsv(result: SimulatorResponse) {
  const headers: Array<keyof SimulatorRootCause> = ['rank', 'entity_type', 'entity_id', 'score', 'evidence', 'recommendation'];
  const rows = (result.summary.root_causes ?? []).map((item) => headers.map((h) => csvEscape(item[h])).join(','));
  downloadText(`${result.session_id}-root-causes.csv`, [headers.join(','), ...rows].join('\n'));
}

export function exportSummaryJson(result: SimulatorResponse) {
  const payload = {
    session_id: result.session_id,
    created_at: result.created_at,
    params: result.params,
    summary: result.summary,
    model: result.model,
    persisted_wafer_count: result.persisted_wafer_count,
    persistence_note: result.persistence_note,
  };
  downloadText(`${result.session_id}-summary.json`, JSON.stringify(payload, null, 2), 'application/json;charset=utf-8');
}
