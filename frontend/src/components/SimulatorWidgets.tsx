import { useEffect, useMemo, useRef } from 'react';
import type { CSSProperties } from 'react';
import type { DefectLabel, SimulatorConfusionCell, WaferMatrix } from '../types';
import { LABELS, labelColor } from '../wafer';
import { risk } from '../utils/format';
import { copy, labelName, type Language } from '../i18n';

const dieGood = '#8bb6df';
const dieBad = '#ef5b57';
const dieSoft = 'rgba(40, 69, 143, .36)';

export function Slider({ label, value, min, max, step, suffix = '', format, onChange }: { label: string; value: number; min: number; max: number; step: number; suffix?: string; format?: (v: number) => string; onChange(v: number): void }) {
  return <label className="range-label"><span>{label}<b>{format ? format(value) : `${value}${suffix}`}</b></span><input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} /></label>;
}

export function MiniKpi({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: 'good' | 'risk' }) {
  return <p className={tone ? `tone-${tone}` : ''}><span>{label}</span><strong>{value}</strong>{sub && <em>{sub}</em>}</p>;
}

export function severityClass(value?: string | null) {
  const v = String(value ?? 'Normal').toLowerCase();
  if (v.includes('critical')) return 'severity-critical';
  if (v.includes('warning')) return 'severity-warning';
  if (v.includes('monitor')) return 'severity-monitor';
  return 'severity-normal';
}

export function WaferCanvas({ matrix, label, size = 96, language = 'en' }: { matrix: WaferMatrix; label: DefectLabel; size?: number; language?: Language }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawWafer(ctx, matrix, size, label);
  }, [matrix, label, size]);
  return <canvas className="wafer-canvas" ref={ref} aria-label={`${language === 'ko' ? '웨이퍼 맵' : 'Wafer map'} ${labelName(label, language)}`} />;
}

export function ConfusionMatrix({ cells, language }: { cells: SimulatorConfusionCell[]; language: Language }) {
  const max = Math.max(1, ...cells.map((cell) => cell.count));
  const lookup = useMemo(() => {
    const map = new Map<string, number>();
    cells.forEach((cell) => map.set(`${cell.true_label}|||${cell.predicted_label}`, cell.count));
    return map;
  }, [cells]);
  const shortLabel = (label: string) => language === 'en' ? label.replace('Edge-', 'E-') : labelName(label, language);
  if (!cells.length) return <p className="muted">{copy[language].widgets.confusionEmpty}</p>;
  return <div className="confusion-matrix" style={{ '--matrix-n': LABELS.length } as CSSProperties}>
    <span className="corner">{copy[language].widgets.truePred}</span>
    {LABELS.map((label) => <span key={`h-${label}`} className="axis-label">{shortLabel(label)}</span>)}
    {LABELS.map((trueLabel) => <div className="confusion-row" key={trueLabel}>
      <span className="axis-label row-label">{shortLabel(trueLabel)}</span>
      {LABELS.map((predLabel) => {
        const count = lookup.get(`${trueLabel}|||${predLabel}`) ?? 0;
        const heat = count / max;
        return <span key={`${trueLabel}-${predLabel}`} title={`${labelName(trueLabel, language)} → ${labelName(predLabel, language)}: ${count}`} className="confusion-cell" style={{ '--heat': heat, '--chip': labelColor(predLabel) } as CSSProperties}>{count || ''}</span>;
      })}
    </div>)}
  </div>;
}

function drawWafer(ctx: CanvasRenderingContext2D, matrix: WaferMatrix, size: number, label: DefectLabel) {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  ctx.clearRect(0, 0, size, size);
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.47;
  const shell = ctx.createRadialGradient(cx * 0.72, cy * 0.66, radius * 0.18, cx, cy, radius);
  shell.addColorStop(0, '#d9e9f8');
  shell.addColorStop(0.72, '#bfd7ee');
  shell.addColorStop(1, '#f0a34a');
  ctx.fillStyle = shell;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 0.985, 0, Math.PI * 2);
  ctx.clip();

  const cell = size / Math.max(rows, cols, 1);
  const hot = label === 'None' ? '#8fa1b7' : dieBad;
  for (let y = 0; y < rows; y += 1) {
    const row = matrix[y];
    for (let x = 0; x < cols; x += 1) {
      const value = row?.[x] ?? 0;
      if (value <= 0) continue;
      if (value >= 1.5) {
        ctx.fillStyle = hot || dieBad;
        ctx.globalAlpha = 0.88;
      } else {
        ctx.fillStyle = dieGood;
        ctx.globalAlpha = 0.70;
      }
      ctx.fillRect(x * cell, y * cell, Math.max(0.8, cell * 0.92), Math.max(0.8, cell * 0.92));
    }
  }
  ctx.globalAlpha = 1;
  const shine = ctx.createLinearGradient(0, 0, size, size);
  shine.addColorStop(0, 'rgba(255,255,255,.24)');
  shine.addColorStop(0.28, 'rgba(255,255,255,.08)');
  shine.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = shine;
  ctx.fillRect(0, 0, size, size);
  ctx.restore();

  ctx.lineWidth = Math.max(1, size * 0.006);
  ctx.strokeStyle = '#c8792f';
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = 'rgba(40, 69, 143, .18)';
  for (const r of [0.18, 0.34, 0.50, 0.66, 0.82]) {
    ctx.beginPath();
    ctx.arc(cx, cy, radius * r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.fillStyle = dieSoft;
  ctx.beginPath();
  ctx.arc(cx, cy - radius * 0.98, Math.max(2, size * 0.014), 0, Math.PI * 2);
  ctx.fill();
}

export function ConfusionLegend({ language }: { language: Language }) {
  return <p className="footnote">{copy[language].widgets.confusionLegend}</p>;
}

export function RiskBadge({ value }: { value?: number | null }) {
  return <span className="risk-badge">R{risk(value)}</span>;
}
