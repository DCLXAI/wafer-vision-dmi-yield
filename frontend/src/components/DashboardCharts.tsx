import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { StatsSummary } from '../types';
import { labelColor } from '../wafer';
import { pct } from '../utils/format';
import { copy, labelName, type Language } from '../i18n';

interface Props {
  language: Language;
  stats: StatsSummary | null;
  history: Array<{ created_at: string; predicted_label: string; confidence: number; inference_ms: number }>;
}

export function DashboardCharts({ language, stats, history }: Props) {
  const c = copy[language].charts;
  const labels = (stats?.label_counts ?? []).map((item) => ({ ...item, displayLabel: labelName(item.label, language) }));
  const trend = [...history].reverse().slice(-20).map((item, index) => ({
    index: index + 1,
    label: labelName(item.predicted_label, language),
    confidence: Number((item.confidence * 100).toFixed(1)),
    latency: Number(item.inference_ms.toFixed(1)),
  }));
  const axis = { fill: '#50627a', fontSize: 11 };
  const tooltip = { background: '#ffffff', border: '1px solid #cfd9e8', borderRadius: 4, color: '#223047', boxShadow: '0 10px 26px rgba(62, 83, 112, .16)' };
  return <section className="charts" id="charts-panel">
    <div className="card chart wide">
      <div className="section-heading"><div><p className="eyebrow">{c.classDist}</p><h2>{c.defectVolume}</h2></div><span className="dim">{stats?.total_predictions ?? 0} {c.predictions}</span></div>
      <div className="chartbox"><ResponsiveContainer width="100%" height="100%"><BarChart data={labels} margin={{ top: 8, right: 12, bottom: 8, left: 0 }}><CartesianGrid stroke="#e3e9f2" vertical={false} /><XAxis dataKey="displayLabel" tick={axis} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} /><YAxis allowDecimals={false} tick={axis} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} /><Tooltip cursor={{ fill: 'rgba(79,121,231,.08)' }} contentStyle={tooltip} /><Bar dataKey="count" radius={[2, 2, 0, 0]}>{labels.map((e) => <Cell key={e.label} fill={labelColor(e.label)} />)}</Bar></BarChart></ResponsiveContainer></div>
    </div>
    <div className="card chart">
      <div className="section-heading"><div><p className="eyebrow">{c.mix}</p><h2>{c.patternShare}</h2></div></div>
      <div className="chartbox small"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={labels} dataKey="count" nameKey="displayLabel" innerRadius={48} outerRadius={78} paddingAngle={3}>{labels.map((e) => <Cell key={e.label} fill={labelColor(e.label)} />)}</Pie><Tooltip contentStyle={tooltip} /></PieChart></ResponsiveContainer></div>
      <p className="muted center">{c.avgConfidence} {pct(stats?.average_confidence)}</p>
    </div>
    <div className="card chart wide">
      <div className="section-heading"><div><p className="eyebrow">{c.recentTrend}</p><h2>{c.confidenceRuns}</h2></div></div>
      <div className="chartbox"><ResponsiveContainer width="100%" height="100%"><LineChart data={trend} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}><CartesianGrid stroke="#e3e9f2" vertical={false} /><XAxis dataKey="index" tick={axis} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} /><YAxis domain={[0, 100]} tick={axis} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} /><Tooltip contentStyle={tooltip} formatter={(v) => [`${v}%`, c.confidence]} /><Line type="monotone" dataKey="confidence" stroke="#36ec83" strokeWidth={2} dot={{ r: 3, fill: '#36ec83', stroke: '#ffffff', strokeWidth: 1 }} activeDot={{ r: 5 }} /></LineChart></ResponsiveContainer></div>
    </div>
  </section>;
}
