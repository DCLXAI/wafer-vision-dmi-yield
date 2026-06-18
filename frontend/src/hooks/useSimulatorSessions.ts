import { useCallback, useEffect, useState } from 'react';
import type { ApiClient, SimulatorSessionListItem } from '../types';

export function useSimulatorSessions(api: ApiClient, pageSize = 8) {
  const [sessions, setSessions] = useState<SimulatorSessionListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);

  const refresh = useCallback(async (nextOffset = offset) => {
    try {
      const page = await api.listSimulationSessions({ limit: pageSize, offset: nextOffset });
      setSessions(page.items);
      setTotal(page.total);
      setOffset(page.offset);
    } catch {
      // Session listing is best-effort; demo mode can still run without it.
    }
  }, [api, offset, pageSize]);

  useEffect(() => { void refresh(0); }, [api, pageSize]);

  const prev = useCallback(() => refresh(Math.max(0, offset - pageSize)), [offset, pageSize, refresh]);
  const next = useCallback(() => refresh(offset + pageSize), [offset, pageSize, refresh]);

  return { sessions, total, offset, pageSize, refresh, prev, next, hasPrev: offset > 0, hasNext: offset + sessions.length < total };
}
