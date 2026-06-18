import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import type { DefectLabel, SimulatorResponse } from '../types';
import { labelColor } from '../wafer';
import { pct, risk } from '../utils/format';
import { copy, labelName, type Language } from '../i18n';

interface ScatterPoint {
  x: number;
  y: number;
  z: number;
  label: DefectLabel | string;
  id: string;
}

const tooltip = {
  background: '#ffffff',
  border: '1px solid #cfd9e8',
  borderRadius: 4,
  color: '#223047',
  boxShadow: '0 10px 26px rgba(62, 83, 112, .16)',
};
const axis = { fill: '#50627a', fontSize: 11 };
const grid = '#e3e9f2';

export function PatternDistributionChart({ result, language }: { result: SimulatorResponse | null; language: Language }) {
  const labels = (result?.summary.label_counts ?? []).map((item) => ({ ...item, displayLabel: labelName(item.label, language) }));
  return <div className="chartbox pro-chart">
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={labels} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
        <CartesianGrid stroke={grid} vertical={false} />
        <XAxis dataKey="displayLabel" tick={{ ...axis, fontSize: 10 }} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} />
        <YAxis tick={{ ...axis, fontSize: 10 }} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} allowDecimals={false} />
        <Tooltip contentStyle={tooltip} />
        <Bar dataKey="count" radius={[2, 2, 0, 0]}>{labels.map((item) => <Cell key={String(item.label)} fill={labelColor(item.label)} />)}</Bar>
      </BarChart>
    </ResponsiveContainer>
  </div>;
}

export function SimulatorAnalyticsCharts({ result, scatter, language }: { result: SimulatorResponse | null; scatter: ScatterPoint[]; language: Language }) {
  const c = copy[language].charts;
  return <div className="lab-grid charts-row pro-charts-row">
    <section className="card simulator-chart wide">
      <div className="section-heading"><div><p className="eyebrow">{c.toolPareto}</p><h3>{c.avgRiskTool}</h3></div></div>
      <div className="chartbox sim-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={result?.summary.tool_risk ?? []} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}><CartesianGrid stroke={grid} vertical={false} /><XAxis dataKey="tool_id" tick={axis} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} /><YAxis tick={axis} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} /><Tooltip contentStyle={tooltip} formatter={(v) => [risk(Number(v)), c.risk]} /><Bar dataKey="avg_risk" fill="var(--risk)" radius={[2, 2, 0, 0]} /></BarChart></ResponsiveContainer></div>
    </section>
    <section className="card simulator-chart wide">
      <div className="section-heading"><div><p className="eyebrow">{c.spc}</p><h3>{c.sequence}</h3></div></div>
      <div className="chartbox sim-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={result?.summary.trend ?? []} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}><CartesianGrid stroke={grid} vertical={false} /><XAxis dataKey="index" tick={axis} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} /><YAxis yAxisId="left" domain={[0, 1]} tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`} tick={axis} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} /><YAxis yAxisId="right" orientation="right" domain={[0, 100]} tick={axis} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} /><Tooltip contentStyle={tooltip} formatter={(v, name) => name === 'yield_estimate' ? [pct(Number(v)), c.yield] : [risk(Number(v)), c.risk]} /><Line yAxisId="left" type="monotone" dataKey="yield_estimate" stroke="#36ec83" strokeWidth={2} dot={false} /><Line yAxisId="right" type="monotone" dataKey="risk_score" stroke="#e84d6f" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></div>
    </section>
    <section className="card simulator-chart">
      <div className="section-heading"><div><p className="eyebrow">{c.confidence}</p><h3>{c.densityConfidence}</h3></div></div>
      <div className="chartbox sim-chart"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}><CartesianGrid stroke={grid} /><XAxis type="number" dataKey="x" name={language === 'ko' ? '결함' : 'defect'} unit="%" tick={axis} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} /><YAxis type="number" dataKey="y" name={c.confidence} unit="%" domain={[40, 100]} tick={axis} axisLine={{ stroke: '#c8d3e2' }} tickLine={false} /><Tooltip contentStyle={tooltip} cursor={{ strokeDasharray: '3 3', stroke: '#8798af' }} /><Scatter data={scatter}>{scatter.map((item) => <Cell key={item.id} fill={labelColor(item.label)} fillOpacity={0.78} stroke="#ffffff" strokeWidth={0.8} />)}</Scatter></ScatterChart></ResponsiveContainer></div>
    </section>
  </div>;
}
