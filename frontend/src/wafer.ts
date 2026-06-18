import type { DefectLabel, WaferMatrix } from './types';

export const LABELS: DefectLabel[] = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'None'];

export const LABEL_INFO: Record<string, { title: string; cause: string; action: string }> = {
  Center: {
    title: 'Center defect',
    cause: '웨이퍼 중앙부에 결함이 집중된 패턴입니다. 노광/식각 균일도, chuck contact, thermal non-uniformity를 의심할 수 있습니다.',
    action: '중앙부 공정 로그와 recipe drift를 우선 비교하세요.',
  },
  Donut: {
    title: 'Donut defect',
    cause: '고리 형태의 결함입니다. 코팅, CMP, deposition profile 또는 radial uniformity 이슈와 연결될 수 있습니다.',
    action: '막 두께 profile과 radial uniformity를 같이 확인하세요.',
  },
  'Edge-Loc': {
    title: 'Edge localized defect',
    cause: '가장자리 특정 구역에 결함이 몰린 패턴입니다. edge handling, clamp, local contamination 가능성이 있습니다.',
    action: '웨이퍼 핸들링 이력과 edge bead removal 조건을 점검하세요.',
  },
  'Edge-Ring': {
    title: 'Edge ring defect',
    cause: '가장자리 링 형태의 결함입니다. plasma edge effect, deposition/etch non-uniformity, edge exclusion 이슈를 의심할 수 있습니다.',
    action: 'edge zone 공정 파라미터와 chamber condition을 먼저 확인하세요.',
  },
  Loc: {
    title: 'Localized defect',
    cause: '국소 영역 결함입니다. particle contamination, reticle issue, handling mark 가능성이 있습니다.',
    action: 'lot 간 공통 좌표와 장비별 발생률을 추적하세요.',
  },
  'Near-full': {
    title: 'Near-full defect',
    cause: '웨이퍼 대부분이 결함으로 분류되는 위험 패턴입니다. 공정 실패나 measurement issue일 수 있습니다.',
    action: '즉시 lot hold, tool health check, upstream 로그 검증이 필요합니다.',
  },
  Random: {
    title: 'Random defect',
    cause: '무작위 결함 분포입니다. 파티클, measurement noise, 일반 수율 저하와 연결될 수 있습니다.',
    action: '시간대/장비/recipe별 defect density trend를 확인하세요.',
  },
  Scratch: {
    title: 'Scratch defect',
    cause: '선형 결함입니다. 이송/접촉/세정/핸들링 과정의 물리적 스크래치 가능성이 큽니다.',
    action: 'robot arm, cassette, cleaning brush, handling path를 점검하세요.',
  },
  None: {
    title: 'No obvious defect pattern',
    cause: '명확한 공간적 불량 패턴이 탐지되지 않은 케이스입니다.',
    action: '패턴은 안정적이지만 die-level defect density는 별도로 확인하세요.',
  },
};

export function labelColor(label: string): string {
  return ({
    Center: '#35aeea',
    Donut: '#aa35e8',
    'Edge-Loc': '#e84d6f',
    'Edge-Ring': '#ffd84d',
    Loc: '#36ec83',
    'Near-full': '#d92b73',
    Random: '#f5b83d',
    Scratch: '#28458f',
    None: '#8fa1b7',
  } as Record<string, string>)[label] ?? '#4f79e7';
}

function inCircle(x: number, y: number, size: number, radius = size * 0.46) {
  const c = (size - 1) / 2;
  return Math.hypot(x - c, y - c) <= radius;
}

export function createPattern(label: DefectLabel, size = 64): WaferMatrix {
  const matrix = Array.from({ length: size }, () => Array.from({ length: size }, () => 0));
  const c = (size - 1) / 2;
  const setDefect = (x: number, y: number) => {
    if (x >= 0 && y >= 0 && x < size && y < size && inCircle(x, y, size)) matrix[y][x] = 2;
  };
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      if (!inCircle(x, y, size)) continue;
      matrix[y][x] = 1;
      const d = Math.hypot(x - c, y - c);
      if (label === 'Center' && d < size * 0.13) setDefect(x, y);
      if (label === 'Donut' && d > size * 0.17 && d < size * 0.28) setDefect(x, y);
      if (label === 'Edge-Ring' && d > size * 0.38 && d < size * 0.46) setDefect(x, y);
      if (label === 'Edge-Loc' && d > size * 0.34 && x > size * 0.62 && y < size * 0.42) setDefect(x, y);
      if (label === 'Loc' && Math.hypot(x - size * 0.34, y - size * 0.67) < size * 0.1) setDefect(x, y);
      if (label === 'Near-full' && d < size * 0.44) setDefect(x, y);
      if (label === 'Scratch' && Math.abs(y - (0.45 * x + size * 0.22)) < 1.5 && x > size * 0.18 && x < size * 0.85) setDefect(x, y);
      if (label === 'Random') {
        const pseudo = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
        if (pseudo - Math.floor(pseudo) > 0.965) setDefect(x, y);
      }
    }
  }
  return matrix;
}

export function matrixStats(matrix: WaferMatrix | null) {
  if (!matrix?.length) return { defectRatio: 0.04, edgeRatio: 0.04, centerRatio: 0.04, scratchScore: 0.02 };
  const h = matrix.length;
  const w = matrix[0].length;
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  let wafer = 0, defects = 0, edge = 0, edgeDefects = 0, center = 0, centerDefects = 0, diagonal = 0;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const value = matrix[y][x];
      if (value <= 0) continue;
      wafer += 1;
      const bad = value >= 1.5;
      if (bad) defects += 1;
      const d = Math.hypot((x - cx) / w, (y - cy) / h);
      if (d > 0.36) { edge += 1; if (bad) edgeDefects += 1; }
      if (d < 0.16) { center += 1; if (bad) centerDefects += 1; }
      if (Math.abs(y - (0.45 * x + h * 0.22)) < 1.8 && bad) diagonal += 1;
    }
  }
  return {
    defectRatio: defects / Math.max(1, wafer),
    edgeRatio: edgeDefects / Math.max(1, edge),
    centerRatio: centerDefects / Math.max(1, center),
    scratchScore: diagonal / Math.max(1, Math.min(w, h)),
  };
}
