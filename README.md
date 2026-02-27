# HaloFinder

A Python implementation of a galaxy-group halo finder pipeline, adapted from earlier work by Kai Wang. The project estimates group membership and halo masses from a galaxy catalog using iterative group-finding, cosmology utilities, luminosity-based ranking, and optional mock-catalog comparison.

## What this repository provides

- A configurable end-to-end halo-finding run (`src/main.py`).
- A `HaloFinder`/`TinkerFinder` workflow that:
  - loads and validates catalog data,
  - computes comoving distances,
  - builds a KD-tree for neighborhood queries,
  - iteratively updates group membership,
  - assigns halo masses from luminosity function and HMF matching,
  - optionally evaluates results against mock truth with bijective matching.
- Reusable science utilities split into focused modules:
  - cosmology (`src/cosmo_funcs.py`),
  - luminosity and HMF utilities (`src/luminosity_funcs.py`),
  - group finding/properties (`src/group_finding_funcs.py`, `src/group_properties_funcs.py`),
  - config parsing (`src/utils.py`),
  - mock scoring (`src/bijective_matching.py`).

---

## Repository layout

```text
halofinder/
├── src/
│   ├── main.py                    # CLI entry point
│   ├── halo_finder.py             # Core finder classes and run pipeline
│   ├── cosmo_funcs.py             # Cosmology distances/conversions
│   ├── luminosity_funcs.py        # LF/HMF utilities and mass updates
│   ├── group_finding_funcs.py     # Membership update logic
│   ├── group_properties_funcs.py  # Group center/luminosity/mass helpers
│   ├── bijective_matching.py      # Mock-vs-measured group matching score
│   ├── halo_p_M_funcs.py          # Halo probability helpers
│   └── utils.py                   # Config reader
├── tests/                         # Unit tests
├── config_sharks.yaml             # Example configuration
└── README.md
```

---

## Requirements

Use Python 3.10+ (recommended) and install dependencies used across the source tree:

- `numpy`
- `scipy`
- `astropy`
- `numba`
- `numba-kdtree`
- `matplotlib`
- `hmf`
- `PyYAML`
- `pytest` (for tests)
- `ruff` (for lint/format)

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install numpy scipy astropy numba numba-kdtree matplotlib hmf PyYAML pytest ruff
```

---

## Configuration

The project is configured via YAML (see `config_sharks.yaml`).

### Key sections

- `run_options`: controls which workflow mode runs.
  - `run_group_finder`
  - `run_mock_comparison`
  - `optimse_on_mock`
  - `optimse_parameter_space`
  - `make_plots`
- `cosmology`: includes `h` and `omega_matter`.
- `survey_fractional_sky_area`: survey sky fraction for volume calculations.
- `column_names`: maps required data fields to input catalog column names.
- `file_locations`: input catalog path and output files/directories.
- `setup_options`: finder thresholds, pivots, SHMR parameters, iteration count, etc.
- `hmf_options`: mass range and resolution for halo mass function generation.
- `luminosity_function_options`: Schechter parameters (`phi_star`, `M_star`, `alpha`).
- `mock_comparison_options`: settings used in bijective matching.

### Input data expectations

Your input table should provide columns mapped in `column_names`, including at minimum:

- galaxy identifier,
- RA/Dec,
- observed redshift,
- absolute magnitude,
- k-correction,
- red/blue flag,
- stellar mass.

If you enable mock-comparison-related modes, a truth/group ID column is required.

---

## Running

From the repository root:

```bash
python src/main.py config_sharks.yaml
```

### Run modes

`main.py` dispatches based on `run_options`:

1. **Single run** (`run_group_finder` or `run_mock_comparison`):
   - executes the finder once,
   - saves outputs,
   - optionally prints `S-score` when comparing to mock groups.

2. **Parameter optimization** (`optimse_on_mock`):
   - optimizes threshold parameters with SciPy L-BFGS-B,
   - uses mock-comparison score as objective.

3. **Grid search** (`optimse_parameter_space`):
   - evaluates parameter combinations,
   - writes results to a CSV derived from `s_tot_path`.

---

## Outputs

Depending on mode and config, outputs may include:

- group assignment table (`galaxy_group_path`),
- mock comparison score output (`s_tot_path` and/or grid CSV),
- plots in `plots_dir` when `make_plots: True`.

---

## Development workflow

### Lint and formatting (PEP 8-focused)

```bash
ruff format src tests
ruff check src tests --select E,W --line-length 120
```

### Run tests

```bash
PYTHONPATH=src pytest -q
```

---

## Notes and caveats

- The current codebase contains domain-specific assumptions tuned for SHARKS-like data and associated column conventions.
- A few config keys intentionally use the legacy spelling `optimse_*` to match existing code paths.
- For large catalogs, enabling plotting can significantly increase runtime.

---

## License

See `LICENSE`.
