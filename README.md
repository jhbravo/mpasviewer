# 🌐 MPAS-Viewer

> **A Python package for efficient visualization of the MPAS-Atmosphere unstructured mesh**

---

## 📄 Publication [![Publication](https://img.shields.io/badge/Publication-SoftwareX-blue)](https://www.sciencedirect.com/science/article/pii/S2352711025004637)

This package has been published in a peer-reviewed software journal:

**Mendez and Temimi. (2026). MPAS-viewer: A Python package for an efficient visualization of the MPAS-atmosphere unstructured mesh**  
SoftwareX.  
🔗 https://www.sciencedirect.com/science/article/pii/S2352711025004637

If you use this package in your research, please consider citing the paper.

---

## 🥈 Award [![Award](https://img.shields.io/badge/Award-Silver-silver)](https://www.linkedin.com/posts/jhbravo_ams2026-activity-7430397403347972096-Khrb)

This tool was selected as one of the award winners, with the presentation, **Python-based visualization of MPAS data in its native unstructured grid** at the Fifth Symposium on Community Modeling and Innovation, during the 106th Annual Meeting of the American Meteorological Society (AMS2026).  
🔗 https://www.linkedin.com/posts/jhbravo_ams2026-activity-7430397403347972096-Khrb

---

## 📖 Project Pythia Cookbook

A dedicated **Project Pythia Cookbook** is available for MPAS-Viewer, featuring interactive tutorials and reproducible examples that demonstrate how to visualize and analyze MPAS data using the library.

The cookbook serves as a practical learning resource for both new and experienced users.

🔗 https://github.com/ProjectPythia/mpasviewer-cookbook

---

## 🎯 Key Features

- ⚡ Fast rendering on native unstructured MPAS mesh
- 🌍 Supports both **global** and **regional** domains
- 📦 Lightweight and **easy to install** (minimal dependencies)
- 📈 Accurate representation of MPAS-A data without resampling
- 🧩 Compatible with NetCDF outputs from MPAS-A
- 💻 Portable across platforms (Linux, macOS, Windows)

---

## 📦 Installation

You can install MPAS-Viewer directly from GitHub:

```bash
uv add "mpasviewer @ https://github.com/jhbravo/mpasviewer.git"
```

```bash
pip install git+https://github.com/jhbravo/mpasviewer.git
```

---

## 🛠️ Prerequisites

To use **MPAS-Viewer**, you’ll need the following Python packages:

- Python **3.10+** [`xarray`](https://docs.xarray.dev/) [`dask`](https://www.dask.org/) [`numpy`](https://numpy.org/) [`matplotlib`](https://matplotlib.org/) [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/)

### Optional:

- [`earthcmap`](https://github.com/jhbravo/earthcmap) — common colormaps for consistent and appropriate visualization
- [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/) — abstract file system layer (e.g. S3, GCS, HTTPS)
- [`zarr`](https://zarr.readthedocs.io/en/stable/) — for chunked, compressed, cloud-optimized data

---

## 🌍 🌎 🌏 Examples of Use

To become familiar with MPAS-Viewer, we recommend starting with the following notebook, which introduces the main features and general workflow of the tool:

- **[general_use.ipynb](examples/00_geneal_use.ipynb)**
  Basic usage example showing how to load MPAS data and generate a simple plot.

If you prefer to explore MPAS-Viewer without installing any software locally—or if you do not have access to MPAS datasets—you can use the following Google Colab notebooks. These examples retrieve data remotely, allowing you to experiment with the tool directly in your browser:

- **[remote_static_global](examples/16_remote_static_global.ipynb)**
  Visualization of MPAS global quasi-uniform mesh (480 km) datasets from THREDDS servers.

- **[remote_out_nyc](examples/17_remote_out_nyc.ipynb)**
  Visualization of MPAS local quasi-uniform mesh (3 km) datasets from THREDDS servers.

- **[remote_out_PR](examples/18_remote_out_PR.ipynb)**
  Visualization of MPAS regional variable-resolution mesh (15–3 km) datasets from THREDDS servers.

For users interested in working with different types of datasets, additional examples are provided below. Although the underlying data are not publicly accessible, these notebooks demonstrate the use of MPAS-Viewer with a variety of model outputs, including MPAS, MPAS-JEDI, CheMPAS, MPAS-Ocean (MPAS-O), MPAS-SeaIce, and other MPAS cores.

- **[projection](examples/01_projection.ipynb)**
  How to visualize MPAS data in different projections.

- **[static](examples/02_static.ipynb)**
  Visualization of standard MPAS invariant variables.

- **[out_diag](examples/03_out.ipynb)**
  Visualization of standard MPAS output variables.

- **[widget](examples/04_widget.ipynb)**
  Interactive visualization using widgets for dynamic variable and time selection.

- **[out_wofs](examples/05_out_wofs.ipynb)**
  Example using WoFS data, including visualization of forecast output on the MPAS mesh.

- **[graf15m](examples/06_graf15m.ipynb)**
  remote access to GRAF ZARR data with 15-minute forecast interval visualization.

- **[graf05m](examples/07_graf05m.ipynb)**
  remote access to GRAF ZARR data with 5-minute high-frequency precipitation forecast.

- **[out_ocean](examples/08_out_ocean.ipynb)**
  Example demonstrating visualization of MPAS-Ocean output.

- **[out_seaice](examples/09_out_seaice.ipynb)**
  Visualization of MPAS-Seaice outputs.

- **[out_chem](examples/10_out_chem.ipynb)**
  Example how to visualize CheMPAS PM2.5 output.

- **[out_mpas-jedi](examples/11_out_mpas-jedi-diag.ipynb)**
  Visualization of MPAS-JEDI diag output fields.

- **[out_mpas-jedi-out](examples/12_out_mpas-jedi-out.ipynb)**
  Visualization of MPAS-JEDI mpasout output fields.

- **[out_mpas-jedi-da_an](examples/13_out_mpas-jedi-da_an.ipynb)**
  Visualization of MPAS-JEDI data assimilation an fields.

- **[out_mpas-jedi-da_bg](examples/14_out_mpas-jedi-da_bg.ipynb)**
  Visualization of MPAS-JEDI data assimilation bg fields.

---

### 🗺️ Basic usage

```python
# Initialize the main object by providing the mesh file (grid or static)
# and the diagnostic output data (single file or directory)
from mpasviewer import scvtmesh

mpasd = scvtmesh(
    grid_file='/path/to/some/file.grid.nc',
    diag_list='/path/to/some/list/of/files/diag'
)

# Load dataset metadata and variable structure
mpasd.dataset()

# (Optional) Compute rain rate from output variables, if applicable
mpasd.rain_rate()


# Load the full dataset (with Dask support if enabled)
dta = mpasd.load()

# Plot a specific variable as a spatial map at a given time index
mpasd.show(dta, var_name='variable', time_index='yyyy-mm-ddTHH')
```
