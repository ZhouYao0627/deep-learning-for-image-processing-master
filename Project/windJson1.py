# 需要安装netCDF4安装包
from netCDF4 import Dataset
import numpy as np
import os
from pandas import Series
import netCDF4
import csv

# navigating to folder location. You may have to change this.
# base_dir = os.path.abspath(os.path.dirname())

# for file in os.listdir(base_dir):
#   if file.endswith("sst.mnmean.nc"):
# dataset = netCDF4.Dataset(base_dir+"/"+file)
dataset = Dataset(r'E:\Project\nc_data\2022-12-16\gfs.t00z.pgrb2.0p25.f001.nc', mode='r', format="NETCDF4")
# 提供所知道的nc文件变量，必须得先知道变量名称
lat = dataset.variables['latitude'][:]
lon = dataset.variables['longitude'][:]
time = dataset.variables['time']
# time_bnds = dataset.variables['time_bnds']
gust = dataset.variables['gust']

# 原例子
#
#         Ice data
#         If you print the variable ice, you'll find:
#           int16 ice(time, zlev, lat, lon)
#           ...
#           current shape = (1, 1, 720, 1440)
#         Thus, we insert time[0], zlev[0], lat[i] and lon[j] into our csv with the corresponding labels
# with open('ice_data.csv', mode='w') as ice_file:
#            ice_writer = csv.writer(ice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#            ice_writer.writerow(['time', 'zlev', 'lat', 'lon', 'ice'])
#
#            for i in range(0,720):
#                print("row " , i+1, " of 720")
#                for j in range(0,1440):
#                    # print(time_var[0] , '\t', zlev[0], '\t', lat[600], '\t', lon[i],'\t', ice[0,0,600,i] )
#                    ice_writer.writerow([time_var[0], zlev[0], lat[i], lon[j], ice[0,0,i,j]])
#            ice_writer.close()


with open('gust.csv', mode='w') as ice_file:
    ice_writer = csv.writer(ice_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    ice_writer.writerow(['time', 'lat', 'lon', 'gust'])
    # 输入经纬度的维数
    for i in range(0, 721):
        print("row ", i + 1, " of 721")
        for j in range(0, 1440):
            # print(time_var[0] , '\t', zlev[0], '\t', lat[600], '\t', lon[i],'\t', ice[0,0,600,i] )
            ice_writer.writerow([time[859], lat[i], lon[j], gust[859, i, j]])
#    ice_writer.close()


# 转换csv格式之前需要编写以下代码获取nc文件的信息
# dst = Dataset(r'E:\Project\nc_data\2022-12-16\gfs.t00z.pgrb2.0p25.f001.nc', mode='r', format="NETCDF4")
# for attr in dst.ncattrs():
#     # 得到.nc文件的信息
#     print('%s: %s' % (attr, dst.getncattr(attr)))

# print("==================1==================")
# 获取nc文件的变量，也就是表头的变量，如经纬度、温度等
# for var in dst.variables:
#     print(var, end=':\n')
#     for attr in dst[var].ncattrs():
#         print('%s: %s' % (attr, dst[var].getncattr(attr)))
#     print()

# print("===================2=================")
# 获取变量的长度，用于读取这些变量
# dims = ['latitude', 'longitude', 'time', 'gust', 'u-gwd', 'v-gwd']
# for dim in dims:
#     print('%s:%s' % (dim, dst.dimensions[dim].size))
