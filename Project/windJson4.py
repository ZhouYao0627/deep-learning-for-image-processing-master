# import xarray as xr
# import json
# import datetime
# import netCDF4 as nc
# import numpy as np
# import os
#
#
# def mkdir(path):
#     folder = os.path.exists(path)
#
#     if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
#         os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
#         print("---  new folder...  ---")
#         print("---  OK  ---")
#     else:
#         print("---  This folder is already have!  ---")
#
#
# def gen_data_dict(data_array, parameterNumber, parameterNumberName):
#     header = {
#         "parameterUnit": "m/s",
#         'parameterNumber': parameterNumber,
#         'parameterNumberName': parameterNumberName,
#         "dx": lon[1] - lon[0],
#         "dy": lat[1] - lat[0],
#         "la1": lat[0],
#         "la2": lat[len(lat) - 1],
#         "nx": len(lat),
#         "ny": len(lon),
#         "lo1": lon[0],
#         "lo2": lon[len(lon) - 1]
#     }
#     nan_to_none = np.fliplr(np.where(np.isnan(data_array), None, data_array))
#     data_list = list(nan_to_none.ravel('F'))
#     return {'header': header, 'data': data_list}
#
#
# if __name__ == '__main__':
#     today = datetime.date.today()  # 2022-12-16
#     file = "E:\\Project\\json\\{}\\windU".format(today)
#     file1 = "E:\\Project\\json\\{}\\windV".format(today)
#     file2 = "E:\\Project\\json\\{}\\windGust".format(today)
#     mkdir(file)
#     mkdir(file1)
#     mkdir(file2)
#
#     for i in range(11):
#         if 0 < i < 10:
#             file_path = r'E:\Project\nc_data\{}\gfs.t00z.pgrb2.0p25.f00{}.nc'.format(today, i)
#             # file_list = glob(file_path)
#             # file_list.sort()
#
#             ds = xr.open_dataset(file_path, decode_times=False)
#             data = nc.Dataset(file_path)
#
#             lon = np.array(data.variables['longitude'])
#             lat = np.array(data.variables['latitude'])
#
#             gust = np.squeeze(ds['gust'][:])
#             parameterNumber0 = 0
#             parameterNumberName1 = 'wind speed'
#             json_gust = gen_data_dict(gust, parameterNumber0, parameterNumberName1)
#             print("json_gust success" + str(i))
#
#             u = np.squeeze(ds['u-gwd'][:])
#             parameterNumber1 = 1
#             parameterNumberName1 = 'U component of wind'
#             json_u = gen_data_dict(u, parameterNumber1, parameterNumberName1)
#             print("json_u success" + str(i))
#
#             v = np.squeeze(ds['v-gwd'][:])
#             parameterNumber2 = 2
#             parameterNumberName2 = 'v component of wind'
#             json_v = gen_data_dict(v, parameterNumber2, parameterNumberName2)
#             print("json_v success" + str(i))
#
#             # json_list = [json_u, json_v]
#
#             name = os.path.basename(file_path)
#             name1 = name.split('.')
#             name2 = name1[-2][1:4]
#
#             with open(file2 + '\\windJsonGust{}.json'.format(name2), 'w') as fp:
#                 json.dump(json_gust, fp)
#             with open(file + '\\windJsonU{}.json'.format(name2), 'w') as fp:
#                 json.dump(json_u, fp)
#             with open(file1 + '\\windJsonV{}.json'.format(name2), 'w') as fp:
#                 json.dump(json_v, fp)
#             print('All Success: ' + str(i))
#         if 10 <= i < 100:
#             file_path = r'E:\Project\nc_data\{}\gfs.t00z.pgrb2.0p25.f0{}.nc'.format(today, i)
#
#             ds = xr.open_dataset(file_path, decode_times=False)
#             data = nc.Dataset(file_path)
#
#             lon = np.array(data.variables['longitude'])
#             lat = np.array(data.variables['latitude'])
#
#             gust = np.squeeze(ds['gust'][:])
#             parameterNumber0 = 0
#             parameterNumberName1 = 'wind speed'
#             json_gust = gen_data_dict(gust, parameterNumber0, parameterNumberName1)
#             print("json_gust success" + str(i))
#
#             u = np.squeeze(ds['u-gwd'][:])
#             parameterNumber1 = 1
#             parameterNumberName1 = 'U component of wind'
#             json_u = gen_data_dict(u, parameterNumber1, parameterNumberName1)
#             print("json_u success" + str(i))
#
#             v = np.squeeze(ds['v-gwd'][:])
#             parameterNumber2 = 2
#             parameterNumberName2 = 'v component of wind'
#             json_v = gen_data_dict(v, parameterNumber2, parameterNumberName2)
#             print("json_v success" + str(i))
#
#             # json_list = [json_u, json_v]
#
#             name = os.path.basename(file_path)
#             name1 = name.split('.')
#             name2 = name1[-2][1:4]
#
#             with open(file2 + '\\windJsonGust{}.json'.format(name2), 'w') as fp:
#                 json.dump(json_gust, fp)
#             with open(file + '\\windJsonU{}.json'.format(name2), 'w') as fp:
#                 json.dump(json_u, fp)
#             with open(file1 + '\\windJsonV{}.json'.format(name2), 'w') as fp:
#                 json.dump(json_v, fp)
#             print('All Success: ' + str(i))
#         if i > 100:
#             file_path = r'E:\Project\nc_data\{}\gfs.t00z.pgrb2.0p25.f{}.nc'.format(today, i)
#
#             ds = xr.open_dataset(file_path, decode_times=False)
#             data = nc.Dataset(file_path)
#
#             lon = np.array(data.variables['longitude'])
#             lat = np.array(data.variables['latitude'])
#
#             gust = np.squeeze(ds['gust'][:])
#             parameterNumber0 = 0
#             parameterNumberName1 = 'wind speed'
#             json_gust = gen_data_dict(gust, parameterNumber0, parameterNumberName1)
#             print("json_gust success" + str(i))
#
#             u = np.squeeze(ds['u-gwd'][:])
#             parameterNumber1 = 1
#             parameterNumberName1 = 'U component of wind'
#             json_u = gen_data_dict(u, parameterNumber1, parameterNumberName1)
#             print("json_u success" + str(i))
#
#             v = np.squeeze(ds['v-gwd'][:])
#             parameterNumber2 = 2
#             parameterNumberName2 = 'v component of wind'
#             json_v = gen_data_dict(v, parameterNumber2, parameterNumberName2)
#             print("json_v success" + str(i))
#
#             # json_list = [json_u, json_v]
#
#             name = os.path.basename(file_path)
#             name1 = name.split('.')
#             name2 = name1[-2][1:4]
#
#             with open(file2 + '\\windJsonGust{}.json'.format(name2), 'w') as fp:
#                 json.dump(json_gust, fp)
#             with open(file + '\\windJsonU{}.json'.format(name2), 'w') as fp:
#                 json.dump(json_u, fp)
#             with open(file1 + '\\windJsonV{}.json'.format(name2), 'w') as fp:
#                 json.dump(json_v, fp)
#             print('All Success: ' + str(i))
#     print("Finish")
# [
#    {
#       "header": {
#           "discipline": 10,
#           "disciplineName": "Oceanographic_products",
#           "center": 0,
#           "centerName": "Ocean Modeling and Observation Laboratory",
#           "significanceOfRT": 0,
#           "significanceOfRTName": "Analysis",
#           "parameterCategory": 1,
#           "parameterCategoryName": "Currents",
#           "parameterNumberName": "U component of wind",
#           "parameterUnit": "m.s-1",
#           "forecastTime": 0,
#           "surface1Type": 160,
#           "surface1TypeName": "Depth below sea level",
#           "surface1Value": 15,
#           "numberPoints": 1038240,
#           "shape": 0,
#           "shapeName": "Earth spherical with radius = 6,367,470 m",
#           "scanMode": 0
#       },
#       "data": []
#    }
# ]

import xarray as xr
import json
import datetime
import netCDF4 as nc
import numpy as np
import os


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  This folder is already have!  ---")


# def gen_data_dict(data_array, parameterNumber, parameterNumberName):
#     header = {
#         "parameterUnit": "m/s",
#         'parameterNumber': parameterNumber,
#         'parameterNumberName': parameterNumberName,
#         "dx": lon[1] - lon[0],
#         "dy": lat[1] - lat[0],
#         "la1": lat[0],
#         "la2": lat[len(lat) - 1],
#         "nx": len(lat),
#         "ny": len(lon),
#         "lo1": lon[0],
#         "lo2": lon[len(lon) - 1]
#     }
#
#     nan_to_none = np.fliplr(np.where(np.isnan(data_array), None, data_array))
#     data_list = list(nan_to_none.ravel('F'))
#     data_list1 = []
#     for i in range(len(data_list)):
#         data_list1[i] = math.round(data_list[i], 2)
#
#     return {'header': header, 'data': data_list1}
def gen_data_dict(u, v):
    header_U = {
        "parameterUnit": "m.s-1",
        "parameterNumber": 2,
        "parameterNumberName": "eastward_wind",
        "dx": 0.25,
        "dy": 0.25,
        "la1": 90.0,
        "la2": -90.0,
        "parameterCategory": 2,
        "parameterCategoryName": "Momentum",
        "lo2": 359.75,
        "nx": 721,
        "ny": 1440,
        "lo1": 0.0
    }

    header_V = {
        "parameterUnit": "m.s-1",
        "parameterNumber": 3,
        "parameterNumberName": "northward_wind",
        "dx": 0.25,
        "dy": 0.25,
        "la1": 90.0,
        "la2": -90.0,
        "parameterCategory": 2,
        "parameterCategoryName": "Momentum",
        "lo2": 359.75,
        "nx": 721,
        "ny": 1440,
        "lo1": 0.0
    }

    # nan_to_none = np.fliplr(np.where(np.isnan(data_array), None, data_array))
    # data_list = list(nan_to_none.ravel('F'))
    # return {'header': header, 'data': data_list1}
    #
    nan_to_none_u = np.fliplr(np.where(np.isnan(u), None, u))
    data_list_u = list(nan_to_none_u.ravel('F'))

    nan_to_none_v = np.fliplr(np.where(np.isnan(v), None, v))
    data_list_v = list(nan_to_none_v.ravel('F'))

    u1 = [i * 100000 for i in data_list_u]
    v1 = [i * 100000 for i in data_list_v]

    data1 = [
        {"header": header_U, "data": u1},
        {"header": header_V, "data": v1}
    ]
    return data1


if __name__ == '__main__':
    today = datetime.date.today()  # 2022-12-22
    file = "E:\\Project\\json\\{}\\windJson".format(today)
    mkdir(file)

    file_path = r'E:\Project\nc_data\2022-12-20\gfs.t00z.pgrb2.0p25.f008.nc'

    ds = xr.open_dataset(file_path, decode_times=False)
    data = nc.Dataset(file_path)

    # lon = np.array(data.variables['longitude'])
    # lat = np.array(data.variables['latitude'])

    # u = np.squeeze(ds['u-gwd'][:])
    # parameterNumber1 = 1
    # parameterNumberName1 = 'U component of wind'
    # json_u = gen_data_dict(u, parameterNumber1, parameterNumberName1)
    # print("json_u success")
    #
    # v = np.squeeze(ds['v-gwd'][:])
    # parameterNumber2 = 2
    # parameterNumberName2 = 'v component of wind'
    # json_v = gen_data_dict(v, parameterNumber2, parameterNumberName2)
    # print("json_v success")

    u = np.squeeze(ds['u-gwd'][:])
    v = np.squeeze(ds['v-gwd'][:])
    json_uv = gen_data_dict(u, v)
    print("jsonUV success")

    # name = os.path.basename(file_path)
    # name1 = name.split('.')
    # name2 = name1[-2][1:4]

    # with open(file + '\\windJsonUV001.json', 'w') as fp:
    #     json.dump(json_uv, fp)
    print('All Success: ')
print("Finish")
