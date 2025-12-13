variable_attrs = {
    "u10": {"long_name": "u component of 10-meter wind", "standard_name": "u10_wind_velocity", "units": "m s-1", "positive": "east",},
    "v10": {"long_name": "v component of 10-meter wind", "standard_name": "v10_wind_velocity", "units": "m s-1", "positive": "north",},
    "uzonal_surface": {"long_name": "u component of wind at midpoint of lowest model layer", "standard_name": "u_surface_component", "units": "m s-1", "positive": "east",},
    "umeridional_surface": {"long_name": "v component of wind at midpoint of lowest model layer", "standard_name": "v_surface_component", "units": "m s-1", "positive": "north",},
    "uzonal_1km": {"long_name": "u component of wind at 1 km AGL", "standard_name": "u_1km_component", "units": "m s-1", "positive": "east",},
    "umeridional_1km": {"long_name": "v component of wind at 1 km AGL", "standard_name": "v_1km_component", "units": "m s-1", "positive": "north",},
    "uzonal_6km": {"long_name": "u component of wind at 6 km AGL", "standard_name": "u_6km_component", "units": "m s-1", "positive": "east",},
    "umeridional_6km": {"long_name": "v component of wind at 6 km AGL", "standard_name": "v_6km_component", "units": "m s-1", "positive": "north",},
    'uzonal_50hPa': {"long_name": "u component of wind at 50 hPa", "standard_name": "u_50hPa_component", "units": "m s-1", "positive": "east",},
    'uzonal_100hPa': {"long_name": "u component of wind at 100 hPa", "standard_name": "u_100hPa_component", "units": "m s-1", "positive": "east",},
    'uzonal_200hPa': {"long_name": "u component of wind at 200 hPa", "standard_name": "u_200hPa_component", "units": "m s-1", "positive": "east",},
    'uzonal_250hPa': {"long_name": "u component of wind at 250 hPa", "standard_name": "u_250hPa_component", "units": "m s-1", "positive": "east",},
    'uzonal_500hPa': {"long_name": "u component of wind at 500 hPa", "standard_name": "u_500hPa_component", "units": "m s-1", "positive": "east",},
    'uzonal_700hPa': {"long_name": "u component of wind at 700 hPa", "standard_name": "u_700hPa_component", "units": "m s-1", "positive": "east",},
    'uzonal_850hPa': {"long_name": "u component of wind at 850 hPa", "standard_name": "u_850hPa_component", "units": "m s-1", "positive": "east",},
    'uzonal_925hPa': {"long_name": "u component of wind at 925 hPa", "standard_name": "u_925hPa_component", "units": "m s-1", "positive": "east",},
    'umeridional_50hPa': {"long_name": "v component of wind at 50 hPa", "standard_name": "v_50hPa_component", "units": "m s-1", "positive": "north",},
    'umeridional_100hPa': {"long_name": "v component of wind at 100 hPa", "standard_name": "v_100hPa_component", "units": "m s-1", "positive": "north",},
    'umeridional_200hPa': {"long_name": "v component of wind at 200 hPa", "standard_name": "v_200hPa_component", "units": "m s-1", "positive": "north",},
    'umeridional_250hPa': {"long_name": "v component of wind at 250 hPa", "standard_name": "v_250hPa_component", "units": "m s-1", "positive": "north",},
    'umeridional_500hPa': {"long_name": "v component of wind at 500 hPa", "standard_name": "v_500hPa_component", "units": "m s-1", "positive": "north",},
    'umeridional_700hPa': {"long_name": "v component of wind at 700 hPa", "standard_name": "v_700hPa_component", "units": "m s-1", "positive": "north",},
    'umeridional_850hPa': {"long_name": "v component of wind at 850 hPa", "standard_name": "v_850hPa_component", "units": "m s-1", "positive": "north",},
    'umeridional_925hPa': {"long_name": "v component of wind at 925 hPa", "standard_name": "v_925hPa_component", "units": "m s-1", "positive": "north",},
    'uzonal': {"long_name": "u component of wind", "standard_name": "u_component", "units": "m s-1", "positive": "east",},
    'umeridional': {"long_name": "v component of wind", "standard_name": "v_component", "units": "m s-1", "positive": "north",},
    'olrtoa': {'units': 'W m^{-2}', 'long_name': 'all-sky top-of-atmosphere outgoing longwave radiation flux',"standard_name": "All-sky TOA OLR flux",},
}

# 'relhum', 'dewpoint', 'temperature', 'height', 'uzonal', 'umeridional', 'w'

# mslp {'units': 'Pa', 'long_name': 'Mean sea-level pressure'}
# t_isobaric {'units': 'K', 'long_name': 'Temperature interpolated to isobaric surfaces defined in t_iso_levels'}
# t_iso_levels {'units': 'Pa', 'long_name': 'Levels for vertical interpolation of temperature to isobaric surfaces'}
# z_isobaric {'units': 'm', 'long_name': 'Height interpolated to isobaric surfaces defined in z_iso_levels'}
# z_iso_levels {'units': 'Pa', 'long_name': 'Levels for vertical interpolation of height to isobaric surfaces'}
# meanT_500_300 {'units': 'K', 'long_name': 'Mean temperature in the 300 hPa - 500 hPa layer'}
# olrtoa {'units': 'W m^{-2}', 'long_name': 'all-sky top-of-atmosphere outgoing longwave radiation flux'}
# rainc {'units': 'mm', 'long_name': 'accumulated convective precipitation'}
# rainnc {'units': 'mm', 'long_name': 'accumulated total grid-scale precipitation'}
# refl10cm_max {'units': 'dBZ', 'long_name': '10 cm maximum radar reflectivity'}
# refl10cm_1km {'units': 'dBZ', 'long_name': 'diagnosed 10 cm radar reflectivity at 1 km AGL'}
# refl10cm_1km_max {'units': 'dBZ', 'long_name': 'maximum diagnosed 10 cm radar reflectivity at 1 km AGL since last output time'}
# precipw {'units': 'kg m^{-2}', 'long_name': 'precipitable water'}
# u10 {'units': 'm s^{-1}', 'long_name': '10-meter zonal wind'}
# v10 {'units': 'm s^{-1}', 'long_name': '10-meter meridional wind'}
# q2 {'units': 'kg kg^{-1}', 'long_name': '2-meter specific humidity'}
# t2m {'units': 'K', 'long_name': '2-meter temperature'}
# th2m {'units': 'K', 'long_name': '2-meter potential temperature'}
# cape {'units': 'J kg^{-1}', 'long_name': 'Convective available potential energy'}
# cin {'units': 'J kg^{-1}', 'long_name': 'Convective inhibition'}
# lcl {'units': 'm', 'long_name': 'Lifted condensation level'}
# lfc {'units': 'm', 'long_name': 'Level of free convection'}
# srh_0_1km {'units': 'm^2 s^{-2}', 'long_name': 'Storm relative helicity, 0-1 km AGL'}
# srh_0_3km {'units': 'm^2 s^{-2}', 'long_name': 'Storm relative helicity, 0-3 km AGL'}
# uzonal_surface {'units': 'm s^{-1}', 'long_name': 'Zonal wind component at midpoint of lowest model layer'}
# uzonal_1km {'units': 'm s^{-1}', 'long_name': 'Zonal wind component at 1 km AGL'}
# uzonal_6km {'units': 'm s^{-1}', 'long_name': 'Zonal wind component at 6 km AGL'}
# umeridional_surface {'units': 'm s^{-1}', 'long_name': 'Meridional wind component at midpoint of lowest model layer'}
# umeridional_1km {'units': 'm s^{-1}', 'long_name': 'Meridional wind component at 1 km AGL'}
# umeridional_6km {'units': 'm s^{-1}', 'long_name': 'Meridional wind component at 6 km AGL'}
# temperature_surface {'units': 'K', 'long_name': 'Temperature at midpoint of lowest model layer'}
# dewpoint_surface {'units': 'K', 'long_name': 'Dewpoint temperature at midpoint of lowest model layer'}
# updraft_helicity_max {'units': 'm^2 s^{-2}', 'long_name': 'Maximum updraft helicity since last output'}
# w_velocity_max {'units': 'm s^{-1}', 'long_name': 'Maximum column w velocity since last output'}
# wind_speed_level1_max {'units': 'm s^{-1}', 'long_name': 'Maximum wind speed in lowest model level since last output'}
# t_oml {'units': 'K', 'long_name': 'ocean mixed layer temperature'}
# t_oml_initial {'units': 'K', 'long_name': 'ocean mixed layer temperature at initial time'}
# t_oml_200m_initial {'units': 'K', 'long_name': 'ocean mixed layer 200 m mean temperature at initial time'}
# h_oml {'units': 'm', 'long_name': 'ocean mixed layer depth'}
# h_oml_initial {'units': 'm', 'long_name': 'Initial depth of ocean mix layer'}
# hu_oml {'units': 'm^2 s^{-1}', 'long_name': 'ocean mixed layer integrated u (zonal velocity)'}
# hv_oml {'units': 'm^2 s^{-1}', 'long_name': 'ocean mixed layer integrated v (meridional velocity)'}



# relhum_50hPa {'units': 'percent', 'long_name': 'Relative humidity vertically interpolated to 50 hPa'}
# relhum_100hPa {'units': 'percent', 'long_name': 'Relative humidity vertically interpolated to 100 hPa'}
# relhum_200hPa {'units': 'percent', 'long_name': 'Relative humidity vertically interpolated to 200 hPa'}
# relhum_250hPa {'units': 'percent', 'long_name': 'Relative humidity vertically interpolated to 250 hPa'}
# relhum_500hPa {'units': 'percent', 'long_name': 'Relative humidity vertically interpolated to 500 hPa'}
# relhum_700hPa {'units': 'percent', 'long_name': 'Relative humidity vertically interpolated to 700 hPa'}
# relhum_850hPa {'units': 'percent', 'long_name': 'Relative humidity vertically interpolated to 850 hPa'}
# relhum_925hPa {'units': 'percent', 'long_name': 'Relative humidity vertically interpolated to 925 hPa'}
# dewpoint_50hPa {'units': 'K', 'long_name': 'Dewpoint temperature vertically interpolated to 50 hPa'}
# dewpoint_100hPa {'units': 'K', 'long_name': 'Dewpoint temperature vertically interpolated to 100 hPa'}
# dewpoint_200hPa {'units': 'K', 'long_name': 'Dewpoint temperature vertically interpolated to 200 hPa'}
# dewpoint_250hPa {'units': 'K', 'long_name': 'Dewpoint temperature vertically interpolated to 250 hPa'}
# dewpoint_500hPa {'units': 'K', 'long_name': 'Dewpoint temperature vertically interpolated to 500 hPa'}
# dewpoint_700hPa {'units': 'K', 'long_name': 'Dewpoint temperature vertically interpolated to 700 hPa'}
# dewpoint_850hPa {'units': 'K', 'long_name': 'Dewpoint temperature vertically interpolated to 850 hPa'}
# dewpoint_925hPa {'units': 'K', 'long_name': 'Dewpoint temperature vertically interpolated to 925 hPa'}
# temperature_50hPa {'units': 'K', 'long_name': 'Temperature vertically interpolated to 50 hPa'}
# temperature_100hPa {'units': 'K', 'long_name': 'Temperature vertically interpolated to 100 hPa'}
# temperature_200hPa {'units': 'K', 'long_name': 'Temperature vertically interpolated to 200 hPa'}
# temperature_250hPa {'units': 'K', 'long_name': 'Temperature vertically interpolated to 250 hPa'}
# temperature_500hPa {'units': 'K', 'long_name': 'Temperature vertically interpolated to 500 hPa'}
# temperature_700hPa {'units': 'K', 'long_name': 'Temperature vertically interpolated to 700 hPa'}
# temperature_850hPa {'units': 'K', 'long_name': 'Temperature vertically interpolated to 850 hPa'}
# temperature_925hPa {'units': 'K', 'long_name': 'Temperature vertically interpolated to 925 hPa'}
# height_50hPa {'units': 'm', 'long_name': 'Geometric height interpolated to 50 hPa'}
# height_100hPa {'units': 'm', 'long_name': 'Geometric height interpolated to 100 hPa'}
# height_200hPa {'units': 'm', 'long_name': 'Geometric height interpolated to 200 hPa'}
# height_250hPa {'units': 'm', 'long_name': 'Geometric height interpolated to 250 hPa'}
# height_500hPa {'units': 'm', 'long_name': 'Geometric height interpolated to 500 hPa'}
# height_700hPa {'units': 'm', 'long_name': 'Geometric height interpolated to 700 hPa'}
# height_850hPa {'units': 'm', 'long_name': 'Geometric height interpolated to 850 hPa'}
# height_925hPa {'units': 'm', 'long_name': 'Geometric height interpolated to 925 hPa'}
# uzonal_50hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed zonal wind at cell centers, vertically interpolated to 50 hPa'}
# uzonal_100hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed zonal wind at cell centers, vertically interpolated to 100 hPa'}
# uzonal_200hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed zonal wind at cell centers, vertically interpolated to 200 hPa'}
# uzonal_250hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed zonal wind at cell centers, vertically interpolated to 250 hPa'}
# uzonal_500hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed zonal wind at cell centers, vertically interpolated to 500 hPa'}
# uzonal_700hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed zonal wind at cell centers, vertically interpolated to 700 hPa'}
# uzonal_850hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed zonal wind at cell centers, vertically interpolated to 850 hPa'}
# uzonal_925hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed zonal wind at cell centers, vertically interpolated to 925 hPa'}
# umeridional_50hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed meridional wind at cell centers, vertically interpolated to 50 hPa'}
# umeridional_100hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed meridional wind at cell centers, vertically interpolated to 100 hPa'}
# umeridional_200hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed meridional wind at cell centers, vertically interpolated to 200 hPa'}
# umeridional_250hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed meridional wind at cell centers, vertically interpolated to 250 hPa'}
# umeridional_500hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed meridional wind at cell centers, vertically interpolated to 500 hPa'}
# umeridional_700hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed meridional wind at cell centers, vertically interpolated to 700 hPa'}
# umeridional_850hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed meridional wind at cell centers, vertically interpolated to 850 hPa'}
# umeridional_925hPa {'units': 'm s^{-1}', 'long_name': 'Reconstructed meridional wind at cell centers, vertically interpolated to 925 hPa'}
# w_50hPa {'units': 'm s^{-1}', 'long_name': 'Vertical velocity vertically interpolated to 50 hPa'}
# w_100hPa {'units': 'm s^{-1}', 'long_name': 'Vertical velocity vertically interpolated to 100 hPa'}
# w_200hPa {'units': 'm s^{-1}', 'long_name': 'Vertical velocity vertically interpolated to 200 hPa'}
# w_250hPa {'units': 'm s^{-1}', 'long_name': 'Vertical velocity vertically interpolated to 250 hPa'}
# w_500hPa {'units': 'm s^{-1}', 'long_name': 'Vertical velocity vertically interpolated to 500 hPa'}
# w_700hPa {'units': 'm s^{-1}', 'long_name': 'Vertical velocity vertically interpolated to 700 hPa'}
# w_850hPa {'units': 'm s^{-1}', 'long_name': 'Vertical velocity vertically interpolated to 850 hPa'}
# w_925hPa {'units': 'm s^{-1}', 'long_name': 'Vertical velocity vertically interpolated to 925 hPa'}
# vorticity_50hPa {'units': 's^{-1}', 'long_name': 'Relative vorticity vertically interpolated to 50 hPa'}
# vorticity_100hPa {'units': 's^{-1}', 'long_name': 'Relative vorticity vertically interpolated to 100 hPa'}
# vorticity_200hPa {'units': 's^{-1}', 'long_name': 'Relative vorticity vertically interpolated to 200 hPa'}
# vorticity_250hPa {'units': 's^{-1}', 'long_name': 'Relative vorticity vertically interpolated to 250 hPa'}
# vorticity_500hPa {'units': 's^{-1}', 'long_name': 'Relative vorticity vertically interpolated to 500 hPa'}
# vorticity_700hPa {'units': 's^{-1}', 'long_name': 'Relative vorticity vertically interpolated to 700 hPa'}
# vorticity_850hPa {'units': 's^{-1}', 'long_name': 'Relative vorticity vertically interpolated to 850 hPa'}
# vorticity_925hPa {'units': 's^{-1}', 'long_name': 'Relative vorticity vertically interpolated to 925 hPa'}