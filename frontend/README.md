# WaferVision — DMI Yield Intelligence Frontend v0.10

React + Vite + TypeScript + Recharts workspace for the WaferVision semiconductor DMI yield intelligence platform.

This frontend turns rare-defect detection, DMI feature engineering, and yield-risk simulation into a deployable portfolio product demo:

- Upload wafer map files and run classification
- Render a die-level wafer heatmap preview
- Show prediction label, confidence, top-k probabilities, model metadata, and latency
- Visualize defect distribution with Recharts bar / pie / line charts
- Display Week 2 SQLite-backed prediction history through the FastAPI API
- Run in frontend-only demo mode when the backend is not available
- Switch the interface between Korean and English from a single language source
- Share a public URL with the current language encoded in the query string
- Publish with Open Graph metadata, manifest, favicon, static-host redirects, and Vercel headers

## v0.10 public deployment polish

- Spotfire-style light workbook UI with functional toolbar, side rail, sheet tabs, search, settings, density, and edit/view controls.
- Korean/English UI copy is centralized in `src/i18n.ts`; charts, simulator labels, root-cause text, and browser metadata follow the selected language.
- Public deploy assets were added: `manifest.webmanifest`, `robots.txt`, `og-wafer.svg`, `_redirects`, and `vercel.json`.
- The app updates `html lang`, document title, description, Open Graph, and Twitter metadata at runtime.
- Demo mode shows a public-safe status strip and uses synthetic/local browser data by default.
- Simulator cockpit now separates `Run preview` from `Save current session`.
- Browser demo simulator is chunked to keep the main thread responsive during large mock lots.
- Dashboard Recharts components are lazy-loaded from `components/DashboardCharts.tsx`.
- Severity tiles use icon, border weight, and color for faster scanning.
- Added 480px mobile polish for narrow phone widths.

## Run locally

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open:

```txt
http://localhost:5173
```

## Default demo mode

`.env.example` starts in mock mode so the public portfolio demo works without a backend:

```bash
VITE_USE_MOCKS=true
```

Mock mode uses browser `localStorage` and synthetic wafer patterns. It is useful for frontend demos only. Do not present mock confidence as real ML performance.

## Internet deployment

The frontend is a static Vite app and can be deployed to Vercel, Netlify, Cloudflare Pages, GitHub Pages, or any static host.

Recommended public demo settings:

```bash
VITE_USE_MOCKS=true
VITE_API_TIMEOUT_MS=15000
```

Recommended live API settings:

```bash
VITE_USE_MOCKS=false
VITE_API_BASE_URL=https://your-api.example.com
VITE_API_TIMEOUT_MS=15000
```

Do not put production secrets in `VITE_API_KEY` for a public static build. Vite embeds `VITE_*` variables into browser JavaScript.

Build command:

```bash
npm run build
```

Output directory:

```txt
dist
```

Vercel can use the included `vercel.json` for SPA rewrites and security/cache headers. Netlify can use `public/_redirects`.

## Connect to Week 2 backend

Start the backend:

```bash
cd ../backend
PYTHONPATH=src uvicorn wafer_vision_api.app:app --reload --host 0.0.0.0 --port 8000
```

Frontend `.env`:

```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCKS=false
VITE_API_TIMEOUT_MS=15000
VITE_API_KEY=
```

`VITE_API_BASE_URL` accepts either `http://localhost:8000` or `http://localhost:8000/api/v1`; the frontend normalizes it internally.

## Upload support

| File type | Browser heatmap preview | FastAPI inference |
|---|---:|---:|
| `.csv` | yes | yes |
| `.npy` | yes, 2D arrays only | yes |
| `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp` | yes | yes |

CSV/NPY values follow the usual wafer-map convention used in this project:

```txt
0 = outside wafer / no die
1 = normal die
2 = defective die
```

## API endpoints used

```txt
GET  /api/v1/health
GET  /api/v1/model
POST /api/v1/predict
POST /api/v1/predict/array
GET  /api/v1/stats/summary
GET  /api/v1/predictions
```

## Project structure

```txt
wafer-vision-week3-frontend/
├─ src/
│  ├─ App.tsx        # dashboard, upload UX, charts, history table
│  ├─ api.ts         # Week 2 FastAPI client
│  ├─ i18n.ts        # Korean/English copy and localized labels
│  ├─ mock.ts        # browser-only mock API for Vercel demos
│  ├─ parsers.ts     # CSV / NPY / image preview parsing
│  ├─ types.ts       # API response contracts
│  ├─ wafer.ts       # labels, explanations, synthetic patterns
│  └─ styles.css     # Spotfire-style workbook UI
├─ tests/
│  └─ parsers.test.ts
├─ docs/
│  └─ WEEK3_FRONTEND.md
├─ .env.example
└─ package.json
```

## Validation

```bash
npm run build
npm test
```

Current package validation:

```txt
build: passed
vitest: 1 file passed, 2 tests passed
note: Recharts is split into a dedicated vendor chunk; dashboard and simulator charts are lazy-loaded behind Suspense/Error Boundary.
```

## Strong interview framing

> “I framed wafer-map analysis as a semiconductor yield problem: severe class imbalance, rare-defect recall, DMI feature reduction, and chamber-level root-cause analysis. The UI is not static: it consumes a real API contract, renders die-level wafer maps, persists prediction history, and exposes simulator telemetry like an actual engineering workflow.”

## v0.5 Spotfire-style simulator cockpit

This frontend now includes `src/SpotfireSimulator.tsx`, a visual analytics cockpit for wafermap pattern recognition simulations.

Features:

- scenario presets: balanced baseline, edge-ring excursion, scratch handling event, noisy mixed lot
- synthetic wafer-lot generation through the real FastAPI simulator or browser mock mode
- saved simulation session list
- wafer-map trellis grid with dynamic filtering
- selected wafer detail view
- tool/chamber risk Pareto
- yield/risk trend chart
- defect-density × confidence scatter plot
- top-k prediction bars and process-cause explanation

The simulator can be run without a backend when `VITE_USE_MOCKS=true`, and against FastAPI when `VITE_USE_MOCKS=false`.


## v0.8 performance and visual refactor

- Simulator actions are now split into **Run preview** and **Save current session**. Preview runs do not write to SQLite/localStorage.
- Browser mock simulation yields between chunks with `requestAnimationFrame`/`setTimeout`, so large mock lots do not freeze the UI before the loading state paints.
- Main dashboard charts moved to `components/DashboardCharts.tsx` and load through React `lazy()`/`Suspense`; simulator charts remain split from the app shell.
- Severity tiles use icon shape and border thickness in addition to color. The redundant Canvas scanline was removed; the visible large-wafer scanline is CSS-only.
- A 480px breakpoint tightens the cockpit for narrow phones.
