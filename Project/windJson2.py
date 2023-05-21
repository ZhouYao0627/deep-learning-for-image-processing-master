# %%
import xarray as xr
import json
from datetime import datetime, timedelta
import numpy as np
from glob import glob


# %%
def gen_data_dict(data_array):
    """
        data_array: an xarray data with two dimensions; 一个有两个维度的xarray数据。
        parameterNumber: the number the same as example data; 与示例数据相同的数字。
        parameterNumberName: the same as example data. 与示例数据相同。

    """
    numberPoints = np.size(data_array)
    parameterNumberName = 'U_component_of_current'
    parameterNumber = 2
    header_dict = {
        'discipline': 10,
        'disciplineName': 'Oceanographic_products',
        'center': 0,
        'centerName': 'Ocean Modeling and Observation Laboratory',
        # 'refTime': reftime,
        'significanceOfRT': 0,
        'significanceOfRTName': 'Analysis',
        'parameterCategory': 1,
        'parameterCategoryName': 'Currents',
        'parameterNumber': parameterNumber,
        'parameterNumberName': parameterNumberName,
        'parameterUnit': 'm.s-1',
        'forecastTime': 0,
        'surface1Type': 160,
        'surface1TypeName': 'Depth below sea level',
        'surface1Value': 15,
        'numberPoints': numberPoints,
        'shape': 0,
        'shapeName': 'Earth spherical with radius = 6,367,470 m',
        'scanMode': 0

    }
    nan_to_none = np.fliplr(np.where(np.isnan(data_array), None, data_array))
    data_list = list(nan_to_none.ravel('F'))
    return {'header': header_dict, 'data': data_list}


# %%
if __name__ == '__main__':
    file_path = r'E:\Project\nc_data\2022-12-16\*.nc'
    file_list = glob(file_path)
    file_list.sort()
    for file_name in file_list:
        ds = xr.open_dataset(file_name, decode_times=False)
        # %% -----------------covert time --------------
        # matlab_datenum = ds['time']
        # file_time = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)
        # file_time = 1970 + matlab_datenum / 365 / 24 / 60 / 60
        # print(file_time)
        # ds['time'] = file_time
        # reftime = file_time
        # step = 10
        depth_layer = 14
        u = np.squeeze(ds['u-gwd'][:])
        # parameterNumberName = 'U_component_of_current'
        # parameterNumber = 2
        json_u = gen_data_dict(u)
        v = np.squeeze(ds['v-gwd'][:])
        # parameterNumberName = 'V_component_o f_current'
        # parameterNumber = 3
        json_v = gen_data_dict(v)
        json_list = [json_u, json_v]
        # date_str = file_time
        if depth_layer == 0:
            surface = 'surface'
        else:
            # surface = f'{depth_layer}m'
            surface = '100m'
        # %% ------------------output-------------------
        with open(f'E:\Project\\json\\2022-12-16\\windJsonTest1.json', 'w') as fp:
            json.dump(json_list, fp)
        # ds2 = xr.combine_by_coords([u,v])
        # ds2.to_netcdf(f'20090101-20090103\ROMS_0_{date_str}.nc')
    print('Done')
