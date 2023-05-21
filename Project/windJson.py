from netCDF4 import Dataset
import numpy as np
import os
from pandas import Series
import netCDF4 as nc
import csv
import xarray as xr

# 打开nc文件，查看文件的纬度信息
path = r'E:\Project\nc_data\2022-12-16\gfs.t00z.pgrb2.0p25.f001.nc'  # 选取一个nc文件路径
# data = xr.open_dataset(path)
data = nc.Dataset(path)
# print("data: ", data)

# 取出维度信息，写入csv
gust = data.variables['gust'][:]  # 风速
ugwd = data.variables['u-gwd'][:]  # 风的 u 分量
vgwd = data.variables['v-gwd'][:]  # 风的 v 分量
longitude = data.variables['longitude'][:]  # 经度
latitude = data.variables['latitude'][:]  # 维度
time = data.variables['time']  # 时间维

with open('windJson.csv', mode='w') as ice_file:
    ice_writer = csv.writer(ice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    ice_writer.writerow(['time', 'latitude', 'longitude', 'gust', 'ugwd', 'vgwd'])  # 依次是列名：时间，纬度，经度，风速，u分量，v分量
    for i in range(0, 721):  # 721为我的纬度数量
        print("row", i + 1, "of 721")  # 打印转换进度
        for j in range(0, 1440):  # 1440为我的经度纬度
            # 写入到csv
            ice_writer.writerow([time, latitude[i], longitude[j], gust[:, i, j], ugwd[:, i, j], vgwd[:, i, j]])
