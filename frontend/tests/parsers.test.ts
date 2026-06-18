import { describe, expect, it } from 'vitest';
import { parseCsv, resizeNearest } from '../src/parsers';

describe('wafer parsers', () => {
  it('parses numeric csv wafer maps', () => {
    expect(parseCsv('0,1,2\n0,1,2')).toEqual([[0, 1, 2], [0, 1, 2]]);
  });

  it('resizes with nearest neighbor', () => {
    const resized = resizeNearest([[0, 1], [2, 1]], 4);
    expect(resized).toHaveLength(4);
    expect(resized[0]).toHaveLength(4);
  });
});
