#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 19:22:37 2025

@author: jhbravo
"""

import os, re
from glob import glob
from datetime import timezone, datetime
from matplotlib.path import Path

import numpy as np
import xarray as xr

import fsspec

from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs

### Static variables
from .statics import variable_attrs

### Earth Color maps
from earthcmap import escmap

class scvtmesh:
    def __init__(self, grid_file, diag_list=None):
        """
        Initialize processor with a required grid or static file and optional diag files.
        
        :param grid_file: (mandatory) Path to the file NetCDF file with the grid, it can be grid or static file.
        :param diag_list: (optional) List of additional NetCDF files.
        """
        self.grid_file_path = grid_file
        self.diag_list = diag_list
        # self.time_dim = "time"
        self.load_variables = None
        self.ds = None
        self.arr = None
        

    def dataset(self, load_variables=None):
        """Load the grid_file NetCDF file."""
        self.load_variables = load_variables
        
        grid_base = xr.open_dataset(self.grid_file_path)

        nEdgesOnCell = grid_base['nEdgesOnCell']
        verticesOnCell = grid_base['verticesOnCell']
        
        def mpas_rad2deg(mpas_file, latlonvar):
            """
            Convert latitude and longitude from radians to degrees.
            The longitude values are between [-180, 180].
            """
            if "lat" in latlonvar:
                return np.rad2deg(mpas_file[latlonvar])
            elif "lon" in latlonvar:
                return (np.rad2deg(mpas_file[latlonvar]) + 180) % 360 - 180
        
        latVertex = mpas_rad2deg(grid_base, 'latVertex')
        lonVertex = mpas_rad2deg(grid_base, 'lonVertex')

        latCell = mpas_rad2deg(grid_base, 'latCell')
        lonCell = mpas_rad2deg(grid_base, 'lonCell')

        latEdge = mpas_rad2deg(grid_base, 'latEdge')
        lonEdge = mpas_rad2deg(grid_base, 'lonEdge')

        mask = np.zeros_like(verticesOnCell.data, dtype=int)
        unique_values = np.unique(nEdgesOnCell.data)

        for n in unique_values:
            mask2 = nEdgesOnCell.data == n
            mask[mask2, 0:n] = 1

        vertOnCell = (verticesOnCell.data * mask) - 1
        
        if not self.diag_list:
            if not set(['ter', 'landmask', 'ivgtyp', 'isltyp']) & set(list(grid_base.keys())):
                ftype = "grid"
            else:
                ftype = "static"
            
            var2map = [x for x in grid_base.keys() if len(grid_base[x].shape) == 1 and 'nCells' in grid_base[x].dims]
            
            outgrid = grid_base.copy()
        else:
            
            ftype = "diag"
            
            # Define the lists
            # ls1 = load_variables#['rainnc', 'rainc', 'precipw', 't2m', "uzonal", 'temperature']
            if self.load_variables != None:
                # This list contains name of variables with height levels
                lvls_vrs = ['relhum', 'dewpoint', 'temperature', 'height', 'uzonal', 'umeridional', 'w']
                
                # Find matching elements in both lists
                matching_items = list(set(load_variables) & set(lvls_vrs))
                
                # Remove matching elements from the original list
                updated_ls1 = [item for item in load_variables if item not in matching_items]
                
                # Create upgraded variables with height levels
                height_levels = ["50hPa", "100hPa", "200hPa", "250hPa", "500hPa", "700hPa", "850hPa", "925hPa"]
                upgraded_items = [f"{var}_{h}" for var in matching_items for h in height_levels]
                
                # Combine the updated list with upgraded variables
                var2map = updated_ls1 + upgraded_items
            
            if os.path.isfile(self.diag_list) and re.match(r".*diag\..*\.nc$", self.diag_list):
                # print("read a single")
                outgrid = xr.open_dataset(self.diag_list)
            elif os.path.isdir(self.diag_list):
                # print("this is to read a glob list")
                allfls = sorted(glob(f"{self.diag_list}/diag*.nc"))
                # outgrid = xr.open_mfdataset(allfls, decode_cf=True, mask_and_scale=False, combine='by_coords')
                outgrid = xr.open_mfdataset(allfls, combine='nested', concat_dim='Time', decode_cf=True, mask_and_scale=False)
            elif re.match(r"^s3://twc-graf-reforecast.*\.zarr$", self.diag_list):
                mapper = fsspec.get_mapper(f"{self.diag_list}/", anon=True)
                outgrid = xr.open_zarr(mapper, consolidated=True)
            else:
                print("not found or doesn't match expected formats")
            
            if var2map:
                outgrid = outgrid[var2map]

            #########################
            # fulvrs = [x for x in outgrid.keys() if len(outgrid[x].shape) == 2 and not 'nVertices' in outgrid[x].dims]
            fulvrs = [x for x in outgrid.keys() if len(outgrid[x].shape) == 2 and 'nCells' in outgrid[x].dims]
            
            # Extract levels from variables
            str_levels = sorted(set(re.findall(r'\d+hPa', " ".join(fulvrs))), key=lambda x: int(x[:-3]))
            num_levels = [int(re.search(r'\d+', level).group()) for level in str_levels]
            
            # Separate variables into groups
            single_vars = [var for var in fulvrs if not re.search(r'\d+hPa', var) and var not in ['initial_time', 'xtime',]]
            self.level_vars = sorted(set(var.split("_")[0] for var in fulvrs if re.search(r'\d+hPa', var)))
            
            # Create the dictionary
            grpvrs = {'single': single_vars, 'levels': self.level_vars}
            # print(grpvrs)
            if len(grpvrs['levels']) != 0:
                ###########################
                for base_var in grpvrs['levels']:
                    # base_var = grpvrs['levels'][0]
                    lev_vars = [f'{base_var}_{n}' for n in str_levels]
                    outgrid[base_var] = xr.concat([outgrid[var] for var in lev_vars], dim=xr.DataArray(num_levels, dims=['pres'], name='pres'))
                    # outgrid[base_var] = outgrid[base_var].assign_coords(levels=num_levels)
                    fixdim = outgrid[base_var].dims[2]##fixed dimension it can be 'nCells' or 'nVertices'
                    outgrid[base_var] = outgrid[base_var].transpose("Time", "pres", fixdim)
                    s = outgrid[base_var].attrs['long_name']
                    outgrid[base_var].attrs['long_name'] = re.sub(r'to \d+ hPa', '', s).strip()
                
                outgrid['pres'].attrs = {'standard_name':'air_pressure',
                                         'long_name':'pressure coordinate of projection',
                                         'units':'hPa',}
            var2map = grpvrs['single'] + grpvrs['levels']
        
        ############################## Creating the correct dataset
        self.ds = xr.Dataset()
        
        frstm = datetime.now(timezone.utc).replace(tzinfo=None)
        tstr = f'{frstm:%Y-%m-%d %H:%M:%S}Z'
        
        self.ds.attrs = {'Conventions':'CF-1.12 UGRID-1.0',
                         'model_name': 'mpas',
                         'core_name': 'atmosphere',
                         'source': 'MPAS',
                         'source_software': 'MPAS-viewer',
                         'title':'por cambiar!!',
                         'date_created': tstr,
                         'date_modified': tstr,}
        
        self.ds["mesh2d"] = int()
        self.ds["mesh2d"].attrs = {
            "cf_role":                "mesh_topology",
            "long_name":              "Topology data of 2D mesh",
            "topology_dimension":     2,
            ######################### 
            "node_coordinates":       "node_x node_y",
            "node_dimension":         "node", 
            ######################### 
            "edge_node_connectivity": "edge_nodes",
            "edge_dimension":         "edge", 
            "edge_coordinates":       "edge_x edge_y",
            ######################### 
            "face_node_connectivity": "face_nodes",
            "face_dimension":         "face",
            "face_coordinates":       "face_x face_y",        
        }
        
        self.ds['projected_coordinate_system'] = int()  # Create an empty variable for coordinate reference system (CRS)
        dprj = {'name': 'latitude_longitude',
                'epsg': 4326,
                'grid_mapping_name': 'latitude_longitude',
                'semi_major_axis': 6378137.0,
                'semi_minor_axis': 6356752.314245179,
                'inverse_flattening': 298.257223563,
                'reference_ellipsoid_name': 'WGS 84',
                'longitude_of_prime_meridian': 0.0,
                'prime_meridian_name': 'Greenwich',
                'geographic_crs_name': 'WGS 84',
                'horizontal_datum_name': 'World Geodetic System 1984',
                'EPSG_code': 'EPSG:4326',
                'proj4_params': '+proj=longlat +datum=WGS84 +no_defs' }
        self.ds['projected_coordinate_system'].attrs = dprj
        
        self.ds = self.ds.assign_coords(
            node_x = ("node", lonVertex.data),
            node_y = ("node", latVertex.data),
            edge_x = ("edge", lonEdge.data),
            edge_y = ("edge", latEdge.data),
            face_x = ("face", lonCell.data),
            face_y = ("face", latCell.data),
            )
        
        self.ds["face_x"].attrs = {'standard_name': 'longitude',
                                   'long_name': 'longitude',
                                   'units': 'degrees_east',
                                   "mesh": "mesh2d",
                                   "location": "face",}
        
        self.ds["face_y"].attrs = {'standard_name': 'latitude',
                                   'long_name': 'latitude',
                                   'units': 'degrees_north',
                                   "mesh": "mesh2d",
                                   "location": "face",}
        
        #### Set time coordinate
        if ftype == "diag":
            if ".zarr" in self.diag_list:
                ## this is for GRAF model
                match = re.search(r'/([0-9]{10}_[0-9]{2})/mpasout_([0-9]+m)\.zarr', self.diag_list)
                if match:
                    timestamp = match.group(1)
                    interval = match.group(2)
                url = f"s3://twc-graf-reforecast/permutations/{timestamp}_{interval}.txt"
                with fsspec.open(url, anon=True, mode='r') as f:
                    # lines = [line.strip() for line in f if line.strip()]
                    ini_time = [datetime.strptime(line.strip(), '%Y-%m-%d_%H.%M.%S') for line in f if line.strip()]
                frstm0 = str(ini_time[0])
                tunits = f"minutes since {frstm0}"
                outgrid = outgrid.sel(Time = range(len(ini_time)))
            else:
                ## this is for local data
                ini_time = outgrid['Time'].data
                if np.issubdtype(ini_time.dtype, np.integer):
                    ini_time = [datetime.strptime(os.path.basename(f), 'diag.%Y-%m-%d_%H.%M.%S.nc') for f in allfls]
                    frstm0 = str(ini_time[0])
                else:
                    frstm0 = str(ini_time[0].astype('datetime64[s]'))
                tunits = f"hours since {frstm0}"
            
            self.ds = self.ds.assign_coords( time = ("time", ini_time) )
            
            self.ds['time'].attrs = {'long_name':'time',
                                     'standard_name':'time',}
            
            self.ds['time'].encoding['units'] = tunits
            
        #### Set pressure coordinate
        if "pres" in outgrid.coords:
            self.ds = self.ds.assign_coords(
                   pres = ("pres", num_levels)
                   )        
            self.ds['pres'].attrs = {'standard_name':'air_pressure',
                                'long_name':'pressure coordinate of projection',
                                'units':'hPa',}
                
        self.ds["face_nodes"] = xr.DataArray(
            data = vertOnCell,
            coords = {
                "face_x": ("face", lonCell.data),
                "face_y": ("face", latCell.data),
            },
            dims = ["face", "nmax_face"],
            attrs = {
                "cf_role": "face_node_connectivity",
                "mesh": "mesh2d",
                "location": "face",
                "long_name": "Mapping from every face to its corner nodes (counterclockwise)",
                "start_index": 0,
                "_FillValue": -1,
            }
        )
        #################################

        for vn in var2map:
            if ftype in ["grid", "static"]:
                fdims = ["face"]
            elif ftype in ["diag"]:
                fdims = ["time","face"]
            if ftype in ["diag"] and "pres" in outgrid[vn].coords:
                fdims = ["time","pres","face"]

            if vn in variable_attrs.keys():
                vattrs = variable_attrs.get(vn, {})
            else:
                vattrs = outgrid[vn].attrs
            
            self.ds[vn] = xr.DataArray(
                data = outgrid[vn].data,
                coords = {
                    "face_x": ("face", lonCell.data),
                    "face_y": ("face", latCell.data),
                },
                dims = fdims,
                attrs = {
                    "mesh": "mesh2d",
                    "location": "face",
                }
            )
            
            self.ds[vn].attrs.update(vattrs)
            self.ds[vn].attrs.update({"grid_mapping": "projected_coordinate_system"})

        print(f"Loaded grid_file dataset: {self.grid_file_path}")

    
    def rain_rate(self):
        """Return the processed Rain Rate Calculation."""
        if self.ds is None:
            raise ValueError("Dataset not loaded. Run load_static() and add_optional_data() first.")
        
        if ".zarr" in self.diag_list:
            rain_1 = 'rain_bucket'
            rain_2 = 'conv_bucket'
        else:
            rain_1 = 'rainnc'
            rain_2 = 'rainc'
                
        if set([rain_1, rain_2]).issubset(list(self.ds.keys())):
            rain1 = 'rainnc_rate'
            self.ds[rain1] = self.ds[rain_1].copy()
            self.ds[rain1].values *= 0
            self.ds[rain1][1:,:].values += self.ds[rain_1][1:,:].values - self.ds[rain_1][:-1,:].values
            self.ds[rain1].attrs['long_name'] = 'Rain Rate total grid-scale precipitation'
            
            rain2 = 'rainc_rate'
            self.ds[rain2] = self.ds[rain_2].copy()
            self.ds[rain2].data *= 0
            self.ds[rain2][1:,:].data = self.ds[rain_2][1:,:].data - self.ds[rain_2][:-1,:].data
            self.ds[rain2].attrs['long_name'] = 'Rain Rate convective precipitation'
            
            rain3 = 'rain_rate'
            self.ds[rain3] = self.ds[rain_2].copy()
            self.ds[rain3].data *= 0
            self.ds[rain3].data = self.ds[rain1].data + self.ds[rain2].data
            self.ds[rain3].attrs['units'] = 'mm'
            self.ds[rain3].attrs['long_name'] = 'Rain Rate (grid-scale + convective) precipitation'

    def load(self):
        """Return the processed xarray Dataset."""
        if self.ds is None:
            raise ValueError("Dataset not loaded. Run load_static() and add_optional_data() first.")
        
        return self.ds

    def crop(self, ds, lon=None, lat=None):
        """Return the processed xarray Dataset."""
        if self.ds is None:
            raise ValueError("Dataset not loaded. Run load_static() and add_optional_data() first.")
        
        # Define vertices of the rectangle (in clockwise or counter-clockwise order)
        vertices = [
            (min(lon), max(lat)),  # Top-left
            (min(lon), min(lat)),  # Bottom-left
            (max(lon), min(lat)),  # Bottom-right
            (max(lon), max(lat)),  # Top-right
            (min(lon), max(lat)),  # Closing point (must match the first point)
            ]
        p = Path(vertices)
        
        points = np.column_stack((self.ds['face_x'].data, self.ds['face_y'].data))
        flags = p.contains_points(points)
        
        face_id = self.ds["face"][flags]
        
        subds = self.ds.sel(face = face_id.values)
        
        return subds
    
    
    def get_array(self):        
        ndt1 = np.ma.masked_object(self.ds["face_nodes"].fillna(-1), -1).astype('int')
        
        lon1 = np.ma.take(self.ds['node_x'].data, ndt1)
        lat1 = np.ma.take(self.ds['node_y'].data, ndt1)
        
        self.arr = np.ma.dstack((lon1, lat1))
        return self.arr
    

    def latlon2cellid(self, lat,lon):
        """Obtaining the Cell number"""
        
        pnt = (lon,lat)
        # print(pnt)
        self.arr = self.get_array()
        dist = np.sqrt(np.sum(np.square(self.arr - pnt), axis=~0))
        
        idcell = np.where(dist == np.min(dist))
        rcell = idcell[0].tolist()
        
        for ncell in rcell:
            poly_path = Path(self.arr[ncell,:,:])
            if poly_path.contains_point(pnt):
                return ncell


    def nearbycells(self, center_cell):
        cell_vrts = np.ma.masked_object(self.ds["face_nodes"][center_cell].data, -1)
        cell_vrts = cell_vrts.compressed()
    
        # Vectorized check: Find rows that contain any of the target values
        rows_with_values = np.ma.any(np.isin(self.ds["face_nodes"], cell_vrts), axis=1)
    
        # Get the row indices
        matching_indices = np.where(rows_with_values)[0]
    
        return matching_indices


    def cbar_adjust(self, cmap_name):
        ### output color bars
        if cmap_name in ['t2m']:
            cmap_name = "temp_ecmwf"
        if cmap_name in ['ssh']:
            cmap_name = "noaa_sst"          
        elif cmap_name in ["rain_tndncy",'rainnc', 'rainc', 'precipw',] or "rain" in cmap_name:
            cmap_name = "nwps_qpe"
        elif cmap_name in ['apcp_bucket','conv_bucket','rain_bucket']:
            cmap_name = "mrms_prec"
        elif cmap_name in ['olrtoa']:
            cmap_name = "cira_ir108"
        elif cmap_name in ["refl10cm_max"]:
            cmap_name = "mrms_cref"
        ### static color bars
        elif cmap_name in ['ter']:
            cmap_name = "ter"
        elif cmap_name in ['isltyp']:
            cmap_name = "soil_comet"
        elif cmap_name in ['ivgtyp']:
            cmap_name = "lalc"
        elif cmap_name in ['landmask']:
            cmap_name = "laoc"
            
        return cmap_name

###################
    def show(self, ds, var_name, time_index=None, crs=None,figsize=None):
        """
        Plot a PolyCollection for a given variable at a specific time index.
        :param var_name: Name of the variable to plot.
        :param time_index: Time index to visualize (default is 0).
        """
        if ds is None:
            raise ValueError("Dataset not loaded. Run load_static() and add_optional_data() first.")
        
        if var_name not in ds.keys():
            raise ValueError(f"Variable '{var_name}' not found in dataset.")

            
        dvar = ds[var_name]
        ## #######
        ndt1 = np.ma.masked_object(ds["face_nodes"].fillna(-1), -1).astype('int')
        
        lon1 = np.ma.take(ds['node_x'].data, ndt1)
        lat1 = np.ma.take(ds['node_y'].data, ndt1)
        
        arr = np.ma.dstack((lon1, lat1))
        
        ## #######
        lon_min, lon_max, lat_min, lat_max = np.ma.min(arr[:,:,0]), np.ma.max(arr[:,:,0]), np.ma.min(arr[:,:,1]), np.ma.max(arr[:,:,1])
        if (lon_max >= 179.5 and lon_min <= -179.5):
            clon = 0
            clat = 0
        else:
            clon = (lon_min + lon_max) / 2
            clat = (lat_min + lat_max) / 2
        
        if crs is None:
            crs = ccrs.Orthographic(central_longitude=clon, central_latitude=clat)
            # print(clon,clat)
        
        #################
        if figsize is None:
            figsize=(6.4, 4.8) ## this is the default size

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': crs})
        plt.close(fig)
        arr_nan = np.where(arr.mask, np.nan, arr.data)
        
        projected = ax.projection.transform_points(ccrs.PlateCarree(), arr_nan[:,:,0], arr_nan[:,:,1])
        
        arr2 = np.ma.array(projected[:,:,:2], mask=arr.mask, copy=False)
        
        crs_name = type(crs).__name__
        # print(crs_name)
        if crs_name not in ['Orthographic','Geostationary'] and (lon_max >= 179.5 and lon_min <= -179.5):
            ################### This part is to identify the polygons at edges
            row_max = arr2[:,:,0].max(axis=1)  # Max per row
            row_min = arr2[:,:,0].min(axis=1)  # Min per row
            row_dff = row_max - row_min
            
            digit_counts = np.floor(np.log10(row_dff.data)).astype(int) + 1
            mask1 = (digit_counts>=digit_counts.max())
            
            idx = np.unique(np.where(mask1))
            
            arr2[idx] = np.nan
            ###################
        ###
        if 'time' in dvar.coords:
            num_steps = len(dvar['time'])  # Number of time steps
            if time_index is None:
                time_index = str(ds['time'].data.min().astype('datetime64[s]'))
        else:
            num_steps = 0
            
        if num_steps == 0:
            vtme = "timeless"
            values = dvar.values.flatten()
        else:
            vtme = str(dvar.sel(time=time_index)['time'].data.astype('datetime64[m]'))
            values = dvar.sel(time=time_index).values.flatten()
            # vtme = str(dvar.isel(time=time_index)['time'].data.astype('datetime64[m]'))
            # values = dvar.isel(time=time_index).values.flatten()
        ############################################ 
        vnme = dvar.attrs.get('long_name', var_name)
        
        #### Get the appropiated colormap 
        if var_name in ['indexToCellID','nEdgesOnCell','areaCell','meshDensity','cellQuality','gridSpacing']:
            cmap = colormaps['Spectral'].resampled(24)
        else:
            var_name = self.cbar_adjust(var_name)
            if "units" in dvar.attrs.keys():
                cmap, norm = escmap(var_name, units = dvar.attrs['units'])
            else:
                cmap, norm = escmap(var_name)
        
        coll = PolyCollection(arr2)
        ### Arrray values
        coll.set_array(values)
        ### Colors 
        coll.set_cmap(cmap)
        coll.set_edgecolor('face')
        if 'norm' in locals():
            coll.set_norm(norm)
        
        # --- Create Figure & Axis ---
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': crs})
        ax.add_collection(coll)
        ax.autoscale_view()
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(vnme+f"\nat {vtme}")
    
        # --- limits ---
        ax.coastlines(linewidth=0.5)
        if (lon_max >= 179.5 and lon_min <= -179.5):
            ax.set_global()
        else:            
            ax.set_extent([lon_min, lon_max, lat_min, lat_max])
            
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.7, color='gray', alpha=0.5, linestyle='--', dms=True)
        gl.top_labels = False
        gl.right_labels = False
    
        # --- Color Bar ---
        pos = ax.get_position()
        # Adding colorbar aligned with the map's height
        cbar_width = 0.02
        cbar_padding = 0.01
        cbar_ax = fig.add_axes([
            pos.x1 + cbar_padding, # left
            pos.y0,                # bottom
            cbar_width,            # width
            pos.height             # height (match the map!)
        ])
        
        fig.colorbar(coll, cax=cbar_ax, label=var_name)