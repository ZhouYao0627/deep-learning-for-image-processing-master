import json
import netCDF4
from netCDF4 import Dataset
from datetime import datetime as DT
import sys
import time
import numpy as np
import netCDF4 as nc
import os

path = r'E:\Project\nc_data\2023-01-02\gfs.t00z.pgrb2.0p25_hr.f001.nc'
data = nc.Dataset(path)
# print(data)
# print("---------------------")

lon = np.array(data.variables['longitude'])
lat = np.array(data.variables['latitude'])
# print(lon)
# print(lat)

# print("----start-------u and v-----start------")
dx = lon[1] - lon[0]  # dx: 0.25
dy = lat[1] - lat[0]  # dy: -0.25
la1 = lat[0]  # la1: 90.0
la2 = lat[len(lat) - 1]  # la2: -90.0
nx = len(lat)  # nx: 721
ny = len(lon)  # ny: 1440
lo1 = lon[0]  # lo1: 0.0
lo2 = lon[len(lon) - 1]  # lo2: 359.75
print("dx:", dx)
print("dy:", dy)
print("la1:", la1)
print("la2:", la2)
print("nx:", nx)
print("ny:", ny)
print("lo1:", lo1)
print("lo2:", lo2)
print("----end-------u and v-----end------")

name = os.path.basename(path)
name = name.split('.')
print(name[-2][1:4])
