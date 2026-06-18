#!/usr/bin/env node
/**
 * Generates a deterministic simulator snapshot from the real FastAPI backend.
 *
 * Usage:
 *   VITE_API_BASE_URL=http://localhost:8000 VITE_API_KEY=... npm run snapshot:mock
 *
 * This keeps demo fixtures aligned with the backend API contract instead of
 * maintaining a large hand-written mock blob in src/mock.ts.
 */
import { writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const rawBase = process.env.VITE_API_BASE_URL || 'http://localhost:8000';
const base = rawBase.endsWith('/api/v1') ? rawBase : `${rawBase.replace(/\/+$/, '')}/api/v1`;
const apiKey = process.env.VITE_API_KEY || '';
const outFile = resolve(process.cwd(), 'src/mock-simulator-snapshot.json');

const payload = {
  wafer_count: Number(process.env.WAFER_COUNT || 48),
  size: 96,
  return_matrix_size: 48,
  downsample_method: 'area',
  random_seed: 20260617,
  scenario: 'edge-ring-excursion',
  pattern_mix: {
    Center: 0.08,
    Donut: 0.04,
    'Edge-Loc': 0.12,
    'Edge-Ring': 0.34,
    Loc: 0.08,
    'Near-full': 0.02,
    Random: 0.14,
    Scratch: 0.10,
    None: 0.08,
  },
  persist: false,
  client_note: 'Generated mock snapshot',
};

const headers = { 'content-type': 'application/json' };
if (apiKey) headers['x-api-key'] = apiKey;

const res = await fetch(`${base}/simulator/run`, {
  method: 'POST',
  headers,
  body: JSON.stringify(payload),
});

if (!res.ok) {
  const body = await res.text();
  throw new Error(`Snapshot generation failed: ${res.status} ${body}`);
}

const snapshot = await res.json();
await mkdir(dirname(outFile), { recursive: true });
await writeFile(outFile, `${JSON.stringify(snapshot, null, 2)}\n`, 'utf8');
console.log(`Wrote ${outFile}`);
