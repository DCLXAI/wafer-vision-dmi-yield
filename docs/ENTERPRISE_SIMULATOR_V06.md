# Enterprise Simulator v0.6 Notes

## Why this upgrade matters

The earlier simulator looked like a polished portfolio dashboard. v0.6 pushes it toward an industrial analytics product:

- It renders wafer trellis maps using Canvas rather than one SVG/React rectangle per die.
- It controls payload size with downsampled matrices.
- It surfaces process-engineering style signals: tool/chamber excursion, root-cause queue, risk bands, P95 risk, and yield loss.
- It separates simulation speed modes from visual fidelity.

## Performance modes

| Mode | Behavior | Best use |
|---|---|---|
| `turbo` | skips model path, returns compact 40px maps | large 500-1,000 wafer demo |
| `balanced` | default, 56px returned maps | portfolio demo and normal use |
| `fidelity` | higher returned map size | selected screenshots / detailed inspection |

## Important API fields

`SimulatorRequest` additions:

```json
{
  "performance_mode": "balanced",
  "return_matrix_size": 56,
  "use_model": true
}
```

`SimulatorSummary` additions:

```json
{
  "p95_risk": 66.2,
  "yield_loss_pp": 7.4,
  "model_agreement": 0.88,
  "simulation_runtime_ms": 212.4,
  "chamber_risk": [],
  "root_causes": []
}
```

## Demo script

1. Start with `edge-ring-excursion`.
2. Set wafer count to 256.
3. Use `balanced` mode.
4. Run simulation.
5. Show executive KPIs: P95 risk, high-risk count, runtime.
6. Open chamber heatmap and click the highest excursion chamber.
7. Filter trellis by high risk.
8. Select a critical wafer.
9. Explain root-cause hint and recommended action.
10. Load a saved analysis session.
