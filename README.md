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

Below are a few basic examples showing how to use **MPAS-Viewer** to load and visualize MPAS data, demonstrating different capabilities. To help you get started with MPAS-Viewer, we provide sample MPAS-A datasets that you can download from the following HydroShare repository [![Dataset](https://img.shields.io/badge/Dataset-HydroShare-blue)](https://www.hydroshare.org/resource/6be754bf29cb488b815810c35f3f0ac9/):

- **[00_general_use.ipynb](examples/00_geneal_use.ipynb)** 
  Basic usage example showing how to load MPAS data and generate a simple plot.

- **[01_projection.ipynb](examples/01_projection.ipynb)** 
  How to visualize MPAS data in different projections.

- **[02_static.ipynb](examples/02_static.ipynb)** 
  Visualization of standard MPAS invariant variables.

- **[03_out.ipynb](examples/03_out.ipynb)** 
  Visualization of standard MPAS output variables.

- **[04_widget.ipynb](examples/04_widget.ipynb)** 
  Interactive visualization using widgets for dynamic variable and time selection.

- **[05_out_wofs.ipynb](examples/05_out_wofs.ipynb)** 
  Example using WoFS data, including visualization of forecast output on the MPAS mesh.

- **[06_graf15m.ipynb](examples/06_graf15m.ipynb)** 
  remote access to GRAF ZARR data with 15-minute forecast interval visualization.

- **[07_graf05m.ipynb](examples/07_graf05m.ipynb)** 
  remote access to GRAF ZARR data with 5-minute high-frequency precipitation forecast.

- **[08_out_ocean.ipynb](examples/08_out_ocean.ipynb)** 
  Example demonstrating visualization of MPAS-Ocean output.

- **[09_out_seaice.ipynb](examples/09_out_seaice.ipynb)** 
  Visualization of MPAS-Seaiceoutputs.

- **[10_out_chem.ipynb](examples/10_out_chem.ipynb)** 
  Example how to visualize MPAS-Chem PM2.5 output.

- **[11_out_mpas-jedi.ipynb](examples/11_out_mpas-jedi.ipynb)** 
  Visualization of MPAS-JEDI output data.

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