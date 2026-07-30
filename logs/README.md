# Run logs

One log per analyst leg, exactly as the run emitted it. These are the audit trail behind
`reports/` and `results/`: what was launched, against which corpus, what it cost, and what
degraded.

| prefix | run |
|---|---|
| `<driver>_<arm>.log` | the Haiku validation board — 11 drivers x 4 arms, `reports/hk/` |
| `sn_*.log` | the Sonnet leak replication — 3 drivers x 2 arms, `reports/sn/` |
| `smoke*.log` | the pre-launch Gate F pilots |

Each log ends with the run's own summary block — record count, degraded count, retry
counts, carried-forward share, and the estimated spend — followed by the two artifact
paths it wrote. That summary is the thing to read first; it is what the QC gate consumes.

**Why these are versioned.** Every number in `docs/results.md` traces to a `reports/` file,
and every `reports/` file traces to one of these. Without them a reader has to take the
run configuration on trust. They are small (~650 KB total) and they make the board
auditable end to end.

**One caveat on cost figures.** `anthropic_client.py` hardcodes list pricing, so any
Sonnet leg's `est_cost_usd` over-reports by roughly 1.5x against the intro rate in force
before 2026-08-31. The Haiku figures are correct as printed.

Scanned for credentials before committing; the logs carry configuration and counts, never
keys or raw API responses.
