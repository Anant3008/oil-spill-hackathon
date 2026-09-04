from opendrift.readers import reader_netCDF_CF_generic
try:
    reader = reader_netCDF_CF_generic.Reader('output/env_forcing.nc')
    print("Variables found:", reader.variables)
    print("Variable mapping:", reader.variable_mapping)
except Exception as e:
    print("Error:", e)
