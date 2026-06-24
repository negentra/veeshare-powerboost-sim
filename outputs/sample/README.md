# Sample outputs

Generated audit evidence is **not committed** to Git. Each run writes a
self-contained package under the directory you pass with `--out-dir`.

## Generate a quick sample (smoke mode, ~1 minute)

```bash
python run.py --mode smoke --out-dir outputs/sample/smoke
```

## Generate audit evidence (single seed)

```bash
python run.py --mode audit --seed 100 --out-dir outputs/sample/seed100
```

## Expected layout per run

```text
<out-dir>/
|-- tables/          CSV metrics (headline_numbers, cluster tables, traceability)
|-- figures/         PNG and SVG charts (fig1–fig4)
|-- logs/
|   |-- run.log
|   |-- run_manifest.json    SHA-256 hashes + provenance
|   `-- validation_report.json
|-- SUMMARY.md
`-- REPORT.html
```

For multi-seed evidence, use `scripts/run_multiseed.ps1` or the equivalent
`--seed` commands documented in `README.md`. Large result archives may be
published separately via GitHub Releases; keep `run_manifest.json` with every
package for reproducibility.