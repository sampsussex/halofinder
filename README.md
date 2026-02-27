# HaloFinder

A Python implementation of an iterative galaxy-group/halo finder inspired by the Yang et al. style workflow, with a Tinker-style halo mass function mapping and mock-catalogue scoring support.

This repository provides:
- A configurable end-to-end pipeline (`TinkerFinder`) for assigning galaxies to groups.
- Cosmology, luminosity, and halo-mass helper functions in modular files under `src/`.
- A YAML-driven configuration system to adapt the same code to different input catalogues.
- Unit tests for core utility/math modules and group/matching helpers.

---

## What the code does

At a high level, the pipeline in `src/main.py`:
1. Loads a YAML config using `ConfigReader`.
2. Builds and runs a `TinkerFinder` instance.
3. Optimizes the grouping parameter `b_threshold` with `scipy.optimize.minimize_scalar`.
4. Uses the resulting score (`S_tot`) to report the best configuration.

Inside `TinkerFinder`, the workflow is:
1. Load catalogue columns (IDs, sky coordinates, redshift, magnitudes).
2. Load luminosity-function parameters.
3. Generate a halo mass function (HMF).
4. Compute comoving distances and Cartesian coordinates.
5. Build a KDTree for neighbour/assignment queries.
6. Initialize central/satellite assignments.
7. Compute galaxy/group luminosities and group centres.
8. Estimate halo masses and iterate group membership updates.
9. Compute a final mock-matching score (`S_tot`).

---

## Repository layout

```text
halofinder/
├── src/
│   ├── main.py                    # CLI entrypoint + b_threshold optimization
│   ├── halo_finder.py             # HaloFinder/TinkerFinder classes and run loop
│   ├── group_finding_funcs.py     # Membership update logic
│   ├── group_properties_funcs.py  # Group centres, luminosity/mass helpers
│   ├── halo_p_M_funcs.py          # Halo probability/mass-related functions
│   ├── luminosity_funcs.py        # LF/HMF and k-correction helpers
│   ├── cosmo_funcs.py             # Cosmology + geometry utility functions
│   ├── bijective_matching.py      # Mock/group matching and score metrics
│   └── utils.py                   # ConfigReader for YAML configs
├── tests/                         # Pytest suite for core modules
├── config_galform.yaml            # Example configuration
└── README.md
```

---

## Installation

> The project currently does not include a pinned dependency lockfile or `pyproject.toml`, so install dependencies manually (or adapt to your environment manager).

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install runtime dependencies

```bash
pip install numpy scipy matplotlib astropy pyyaml numba-kdtree hmf
```

### 3) Install test dependencies

```bash
pip install pytest
```

---

## Configuration

The code is configured through a YAML file (example: `config_galform.yaml`) with sections for:
- `run_options`
- `cosmology`
- `survey_fractional_sky_area`
- `column_names`
- `file_locations`
- `setup_options`
- `hmf_options`
- `mock_comparison_options`

### Important notes

- `file_locations.galaxy_catalog_path` in the example config points to a machine-specific absolute path. You **must** update it to a local path before running.
- Ensure the `column_names` mapping matches the exact column names in your input table/parquet/fits file.
- `plots_dir` should exist (or be creatable) before execution.

---

## Running the pipeline

From repository root:

```bash
python src/main.py config_galform.yaml
```

Expected behavior:
- The program loads configuration and checks whether group finding is enabled.
- It evaluates different `b_threshold` values using bounded scalar optimization.
- It prints the best continuous `b_threshold` and associated score.

---

## Running tests

From repository root:

```bash
pytest -q
```

The test suite covers:
- Cosmology distance/volume/angle helpers.
- Luminosity/magnitude transforms.
- Group property helper functions.
- Halo-mass helper functions.
- Bijective matching/scoring utilities.
- Configuration loading/error handling.

---

## Outputs and artifacts

Depending on configuration and run settings, the pipeline can produce:
- Group assignment catalogs.
- `S_tot` score outputs for mock comparison.
- Diagnostic plots (luminosity, distances, halo distributions, etc.).

Output locations are controlled via `file_locations` in the YAML config.

---

## Development notes

- Source files are organized as importable modules under `src/`.
- The codebase mixes scientific Python numerics with astrophysical domain assumptions (magnitude limits, HMF range, survey area fractions).
- Logging is enabled in the halo finder workflow and can be extended for deeper diagnostics.

---

## License

This project is distributed under the terms in `LICENSE`.
