import netCDF4
from netCDF4 import Dataset

# 查看nc文件内容
nc_obj = Dataset(r'E:\Project\nc_data\2022-12-16\gfs.t00z.pgrb2.0p25.f001.nc')
# 查看nc文件中的变量，结果是：dict_keys(['time', 'step', 'surface', 'latitude', 'longitude', 'valid_time', 'gust', 'sp', 't', 'heightAboveGround', 't2m', 'r2', 'tp', 'u-gwd', 'v-gwd', 'lowCloudBottom', 'pres'])
keys = nc_obj.variables.keys()
print("keys: ", keys)
# 查看变量的信息 结果是：<class 'netCDF4._netCDF4.Variable'>...
print("nc_obj.variables['time']: ", nc_obj.variables['time'])
# 查看变量的属性 ['long_name', 'standard_name', 'units', 'calendar']
print("nc_obj.variables['time'].ncattrs(): ", nc_obj.variables['time'].ncattrs())
# 查看nc文件中变量数据，结果是：1671148800
print("nc_obj.variables['time'][:]: ", nc_obj.variables['time'][:])

#list 转json
json.dumps(result, separators=(',', ':'))

