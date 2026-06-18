# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

**mpasviewer** is a Python package for visualizing MPAS-Atmosphere (Model for Prediction Across Scales) unstructured mesh data directly on native Voronoi cells — without resampling to a regular grid. It supports MPAS-A, MPAS-Ocean, MPAS-Seaice, MPAS-Chem, and MPAS-JEDI outputs.

## Build & Development Commands

This project uses [uv](https://astral-sh.github.io/uv/) as the package manager.

```bash
uv build          # Build wheel and source distribution
uv publish        # Publish to PyPI (requires auth)
jupyter notebook  # Run examples in examples/
```

Install from source:
```bash
pip install git+https://github.com/jhbravo/mpasviewer.git
```

There is no formal test suite — validation is done via Jupyter notebooks in `examples/`.

## Architecture

### Core Class: `scvtmesh` (`src/mpasviewer/main.py`)

Everything flows through one class. The typical usage pattern:

```python
mpasd = scvtmesh(grid_file='...grid.nc', diag_list='...diag*.nc')
mpasd.dataset()          # Load + parse mesh topology and data
mpasd.rain_rate()        # Optional: compute derived variables
dta = mpasd.load()       # Return processed xarray Dataset
mpasd.show(dta, var_name='temperature', time_index='2024-01-01T00')
```

**Data flow:**
1. `__init__` — stores file paths
2. `dataset()` — reads NetCDF/ZARR, converts lat/lon from radians to degrees, extracts mesh connectivity (cells/vertices/edges), merges multi-level pressure variables (e.g., `temperature_500hPa`)
3. Derived variable methods (`rain_rate()`, `wind_sd()`) — compute and attach new variables
4. `load()` — returns the assembled xarray Dataset (CF 1.12 / UGRID 1.0 compliant)
5. `collection()` / `show()` — builds matplotlib `PolyCollection` from mesh vertices and renders with Cartopy

### Key Design Decisions

- **No interpolation:** Renders directly on MPAS Voronoi polygons via `matplotlib.collections.PolyCollection`. `get_struck()` assembles vertex coordinate arrays from mesh topology.
- **Lazy loading:** Dask is used for large datasets.
- **Multi-source input:** `dataset()` accepts local NetCDF files/directories, S3/GCS (via fsspec), ZARR stores, and THREDDS catalogs (`get_thredds_list()`).
- **CF/UGRID compliance:** Output datasets use standard dimension names (`nCells`, `nVertices`, `nEdges`, `time`, `nPresLevels`) and WGS84/EPSG:4326 CRS metadata.

### `statics.py`

Defines a dictionary of CF-compliant variable attributes (units, long names, colormaps, ranges) for ~60+ MPAS variables including multi-level isobaric fields (50–925 hPa). When adding new variable support, add metadata here.

### `__init__.py`

Exports only `scvtmesh` and `main`. The package entry point `mpasviewer` CLI maps to `mpasviewer:main`.

## Input File Types

| File type | Contains |
|-----------|----------|
| `*.grid.nc` | Mesh topology (connectivity, coordinates) — always required |
| `*.static.nc` | Static invariant fields (terrain, soil, vegetation) |
| `diag.YYYY-MM-DD_HH.MM.SS.nc` | Time-stepped diagnostic output |
| ZARR stores | Cloud-optimized format, accessed via fsspec (S3/GCS) |

## CI/CD

`.github/workflows/publish.yml` triggers on version tags (`v*`), builds with uv, and publishes to PyPI using OIDC trusted publishing (no stored secrets). Smoke tests are currently commented out in the workflow.
