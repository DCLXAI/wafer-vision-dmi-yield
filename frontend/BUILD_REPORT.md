# Build Report

Generated for WaferVision public deployment frontend.

## Commands run

```bash
npm run build
npm test
```

## Result

```txt
npm run build: passed
npm test: passed
Vitest: 1 test file passed, 2 tests passed
```

## Note

Recharts is split into a dedicated vendor chunk. Public deploy assets are copied into `dist`: manifest, robots, Open Graph image, and static-host redirects.
