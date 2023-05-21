# -*- coding: utf-8 -*-
import json
import netCDF4
from netCDF4 import Dataset
from datetime import datetime as DT
import sys
import time


def readNCFile(filename):
    result = []
    nc_obj = Dataset(filename)
    keys = nc_obj.variables.keys()
    # dict_keys(['time', 'step', 'surface', 'latitude', 'longitude', 'valid_time', 'gust', 'sp', 't', 'heightAboveGround', 't2m', 'r2', 'tp', 'u-gwd', 'v-gwd', 'lowCloudBottom', 'pres'])
    # print("keys: ", keys)
    if len(keys) > 0:
        stime = nc_obj.variables['time'][:]
        dtime = time.localtime(stime)
        time_new = time.strftime("%Y-%m-%d %H:%M:%S", dtime)
        dd = {}
        for i in keys:
            if i == 'time':
                dd[i] = time_new
            else:
                dd[i] = str(nc_obj.variables[i][:])
        result.append(dd)
        return json.dumps(result, separators=(',', ':'))
    else:
        return []


if __name__ == '__main__':
    filename = r'E:\Project\nc_data\2022-12-16\gfs.t00z.pgrb2.0p25.f001.nc'
    result = readNCFile(filename)
    with open(f'E:\Project\\json\\2022-12-16\\windJson001', 'w') as fp:
        json.dump(result, fp)
    print(result)

# import time
# stime = 1640693698.2191172
# dtime = time.localtime(stime)
# >>> dtime
# time.struct_time(tm_year=2021, tm_mon=12, tm_mday=28, tm_hour=20, tm_min=14, tm_sec=58, tm_wday=1, tm_yday=362, tm_isdst=0)
# >>> print(time.strftime("%Y-%m-%d %H:%M:%S", dtime))
# 2021-12-28 20:14:58
