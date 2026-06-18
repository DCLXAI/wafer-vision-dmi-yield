import type { Language } from '../i18n';
import { localeFor } from '../i18n';

export const pct = (value?: number | null, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value) ? '—' : `${(value * 100).toFixed(digits)}%`;

export const pp = (value?: number | null) =>
  value === null || value === undefined || Number.isNaN(value) ? '—' : `${value.toFixed(1)}pp`;

export const risk = (value?: number | null) =>
  value === null || value === undefined || Number.isNaN(value) ? '—' : value.toFixed(1);

export const ms = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return value < 1000 ? `${value.toFixed(value < 10 ? 1 : 0)} ms` : `${(value / 1000).toFixed(2)} s`;
};

export const dt = (value: string, language: Language = 'en') =>
  new Intl.DateTimeFormat(localeFor[language], { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value));

export const compact = (value?: number | null, language: Language = 'en') =>
  value === null || value === undefined
    ? '—'
    : new Intl.NumberFormat(localeFor[language], { notation: value > 9999 ? 'compact' : 'standard' }).format(value);
