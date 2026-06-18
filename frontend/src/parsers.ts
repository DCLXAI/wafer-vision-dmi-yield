import type { WaferMatrix } from './types';

export function validateMatrix(matrix: WaferMatrix) {
  if (!matrix.length) throw new Error('Wafer matrix is empty.');
  const width = matrix[0]?.length ?? 0;
  if (!width) throw new Error('Wafer matrix contains an empty row.');
  for (const row of matrix) {
    if (row.length !== width) throw new Error('Wafer matrix must be rectangular.');
    for (const value of row) {
      if (!Number.isFinite(value)) throw new Error('Wafer matrix contains non-finite values.');
    }
  }
}

export function parseCsv(text: string): WaferMatrix {
  const matrix = text.trim().split(/\r?\n/).map((line) =>
    line.trim().replace(/^\[/, '').replace(/\]$/, '').split(/[\s,;]+/).filter(Boolean).map(Number),
  ).filter((row) => row.length > 0);
  validateMatrix(matrix);
  return matrix;
}

export function resizeNearest(matrix: WaferMatrix, target = 64): WaferMatrix {
  validateMatrix(matrix);
  const h = matrix.length;
  const w = matrix[0].length;
  return Array.from({ length: target }, (_, y) => {
    const sy = Math.min(h - 1, Math.floor((y / target) * h));
    return Array.from({ length: target }, (_, x) => {
      const sx = Math.min(w - 1, Math.floor((x / target) * w));
      return matrix[sy][sx];
    });
  });
}

const decoder = new TextDecoder('latin1');

export function parseNpy(buffer: ArrayBuffer): WaferMatrix {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.slice(0, 6));
  if (magic !== '\x93NUMPY') throw new Error('Invalid NPY file.');
  const major = bytes[6];
  const view = new DataView(buffer);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const header = decoder.decode(bytes.slice(headerStart, headerStart + headerLen));
  const descr = /'descr'\s*:\s*'([^']+)'/.exec(header)?.[1];
  const shapeRaw = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)?.[1];
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(header)?.[1] === 'True';
  if (!descr || !shapeRaw) throw new Error('Unsupported NPY header.');
  if (fortran) throw new Error('Fortran-order NPY preview is not supported.');
  const shape = shapeRaw.split(',').map((v) => v.trim()).filter(Boolean).map(Number);
  if (shape.length !== 2) throw new Error(`Only 2D NPY arrays are supported. Got ${shapeRaw}.`);
  const [rows, cols] = shape;
  const dataOffset = headerStart + headerLen;
  const data = new DataView(buffer, dataOffset);
  const little = descr.startsWith('<') || descr.startsWith('|');
  const dtype = descr.replace(/[<>=|]/g, '');
  const read = (i: number) => {
    if (dtype === 'u1') return data.getUint8(i);
    if (dtype === 'i1') return data.getInt8(i);
    if (dtype === 'u2') return data.getUint16(i * 2, little);
    if (dtype === 'i2') return data.getInt16(i * 2, little);
    if (dtype === 'u4') return data.getUint32(i * 4, little);
    if (dtype === 'i4') return data.getInt32(i * 4, little);
    if (dtype === 'f4') return data.getFloat32(i * 4, little);
    if (dtype === 'f8') return data.getFloat64(i * 8, little);
    throw new Error(`Unsupported NPY dtype: ${descr}`);
  };
  return Array.from({ length: rows }, (_, y) => Array.from({ length: cols }, (_, x) => read(y * cols + x)));
}

export async function imageToMatrix(file: File, target = 64): Promise<WaferMatrix> {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement('canvas');
  canvas.width = target;
  canvas.height = target;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas not available.');
  ctx.drawImage(bitmap, 0, 0, target, target);
  const pixels = ctx.getImageData(0, 0, target, target).data;
  const matrix: WaferMatrix = [];
  for (let y = 0; y < target; y += 1) {
    const row: number[] = [];
    for (let x = 0; x < target; x += 1) {
      const i = (y * target + x) * 4;
      const [r, g, b, a] = [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]];
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      const c = (target - 1) / 2;
      const inWafer = Math.hypot(x - c, y - c) <= target * 0.47;
      if (a < 16 || !inWafer || gray < 15) row.push(0);
      else if (gray < 120 || r > g + 40 || r > b + 40) row.push(2);
      else row.push(1);
    }
    matrix.push(row);
  }
  bitmap.close?.();
  return matrix;
}

export async function previewFile(file: File): Promise<{ matrix: WaferMatrix | null; warning?: string }> {
  const name = file.name.toLowerCase();
  try {
    if (name.endsWith('.csv') || file.type.includes('csv')) return { matrix: resizeNearest(parseCsv(await file.text()), 64) };
    if (name.endsWith('.npy')) return { matrix: resizeNearest(parseNpy(await file.arrayBuffer()), 64) };
    if (file.type.startsWith('image/') || /\.(png|jpe?g|bmp|webp)$/i.test(file.name)) return { matrix: await imageToMatrix(file) };
    return { matrix: null, warning: 'Preview supports CSV, NPY, PNG, JPG, BMP, and WEBP.' };
  } catch (error) {
    return { matrix: null, warning: error instanceof Error ? error.message : 'Could not parse preview.' };
  }
}
