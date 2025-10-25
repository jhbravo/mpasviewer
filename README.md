# 🌐 MPAS-Viewer

> **A Python package for efficient visualization of the MPAS-Atmosphere unstructured mesh**

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

## 🛠️ Prerequisites

To use **MPAS-Viewer**, you’ll need the following Python packages:

- Python **3.10+**
- [`xarray`](https://docs.xarray.dev/)
- [`dask`](https://www.dask.org/)
- [`numpy`](https://numpy.org/)
- [`matplotlib`](https://matplotlib.org/)
- [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/)

### Optional (for remote access or cloud workflows):
- [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/) — abstract file system layer (e.g. S3, GCS, HTTPS)
- [`zarr`](https://zarr.readthedocs.io/en/stable/) — for chunked, compressed, cloud-optimized data



---

## 🧪 Examples of Use

Below are a few basic examples showing how to use `MPAS-Viewer` to load and visualize MPAS-Atmosphere data.

---

### 📈 Example 1: Load and Plot a Variable

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
mpasd.show(dta, var_name='refl10cm_max', time_index='2021-09-02T00')
```
