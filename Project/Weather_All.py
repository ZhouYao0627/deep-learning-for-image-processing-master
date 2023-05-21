import requests
import os
import cfgrib
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib
import cartopy.crs as ccrs
from PIL import Image
import datetime
import json


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  This folder is already have!  ---")


def downloaddata(time, file_download):
    try:
        if time == 0:
            print("提示：000号数据无需下载，下方紧跟着一行下载失败，无需在意")
        if 0 < time < 10:
            res = requests.get(
                'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_low_cloud_bottom_level=on&lev_surface=on&var_APCP=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos'
                    .format(time, str(today1)))
        elif 10 <= time < 100:
            res = requests.get(
                'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f0{}&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_low_cloud_bottom_level=on&lev_surface=on&var_APCP=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos'
                    .format(time, str(today1)))
        elif time >= 100:
            res = requests.get(
                'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f{}&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_low_cloud_bottom_level=on&lev_surface=on&var_APCP=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos'
                    .format(time, str(today1)))

        if res.status_code == 200:
            if 0 < time < 10:
                localpath = file_download + "gfs.t00z.pgrb2.0p25.f00{}".format(time)
            elif 10 <= time < 100:
                localpath = file_download + 'gfs.t00z.pgrb2.0p25.f0{}'.format(time)
            elif time >= 100:
                localpath = file_download + 'gfs.t00z.pgrb2.0p25.f{}'.format(time)
            print('contents of URL written to ' + localpath)
            open(localpath, 'wb').write(res.content)
        print("下载成功")
    except:
        print("下载失败")


def change_gfs_to_nc(time, file_open_gfs, file_change_to_nc):
    for i in range(time):
        if 0 < i < 10:
            data = xr.open_dataset(file_open_gfs + '\\gfs.t00z.pgrb2.0p25.f00{}'.format(i), engine='cfgrib')
            data.to_netcdf(file_change_to_nc + '\\gfs.t00z.pgrb2.0p25.f00{}.nc'.format(i))
            print('success change: ' + str(i))
        elif 10 <= i < 100:
            data = xr.open_dataset(file_open_gfs + '\\gfs.t00z.pgrb2.0p25.f0{}'.format(i), engine='cfgrib')
            data.to_netcdf(file_change_to_nc + '\\gfs.t00z.pgrb2.0p25.f0{}.nc'.format(i))
            print('success change: ' + str(i))
        elif i >= 100:
            data = xr.open_dataset(file_open_gfs + '\\gfs.t00z.pgrb2.0p25.f{}'.format(i), engine='cfgrib')
            data.to_netcdf(file_change_to_nc + '\\gfs.t00z.pgrb2.0p25.f{}.nc'.format(i))
            print('success change: ' + str(i))


def ROTATE(path):
    image = Image.open(path)
    out = image.transpose(Image.Transpose.ROTATE_270)  # 逆时针旋转270度
    out.save(path)


def creatimg(time, file_temp_nomal, file_pres_nomal, file_rh_nomal, file_tp_nomal):
    matplotlib.use('Agg')

    plt.rcParams.update({'font.size': 20})
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    for i in range(time):
        if 0 < i < 10:
            # file_path = r'E:\Project\nc_data\2022-12-15\gfs.t00z.pgrb2.0p25_1hr.f008.nc'
            file_path = r'E:\Project\nc_data\{}\gfs.t00z.pgrb2.0p25.f00{}.nc'.format(today, i)
            ds = xr.open_dataset(file_path)

            temp = ds['t'][:].T  # 温度：t->Temperature
            pres = ds['pres'][:].T  # 压力：pres->Pressure
            rh = ds['r2'][:].T  # 相对湿度：rh->2 metre relative humidity
            tp = ds['tp'][:].T  # 降水：tp->Total Precipitation

            level_temp = np.arange(218.1, 331.9, 1)
            level_pres = np.arange(65132.1, 104185.7, 10000)
            level_rh = np.arange(4.2, 100, 1)
            level_tp = np.arange(0, 79.1, 1)
            proj = ccrs.Mercator(central_longitude=179.9, min_latitude=-80, max_latitude=84)

            file_temp = file_temp_nomal + "\\gfs.t00z.pgrb2.0p25.f00{}_temp.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(temp, levels=level_temp, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_temp, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_temp)
            print("success createImg temp" + str(i))

            file_pres = file_pres_nomal + "\\gfs.t00z.pgrb2.0p25.f00{}_pres.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(pres, levels=level_pres, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_pres, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_pres)
            print("success createImg pres" + str(i))

            file_rh = file_rh_nomal + "\\gfs.t00z.pgrb2.0p25.f00{}_rh.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(rh, levels=level_rh, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_rh, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_rh)
            print("success createImg rh" + str(i))

            file_tp = file_tp_nomal + "\\gfs.t00z.pgrb2.0p25.f00{}_tp.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(tp, levels=level_tp, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_tp, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_tp)
            print("success createImg tp" + str(i))
        elif 10 <= i < 100:
            # file_path = r'E:\Project\nc_data\2022-12-15\gfs.t00z.pgrb2.0p25_1hr.f008.nc'
            file_path = r'E:\Project\nc_data\{}\gfs.t00z.pgrb2.0p25.f0{}.nc'.format(today, i)
            ds = xr.open_dataset(file_path)

            temp = ds['t'][:].T  # 温度：t->Temperature
            pres = ds['pres'][:].T  # 压力：pres->Pressure
            rh = ds['r2'][:].T  # 相对湿度：rh->2 metre relative humidity
            tp = ds['tp'][:].T  # 降水：tp->Total Precipitation

            level_temp = np.arange(218.1, 331.9, 1)
            level_pres = np.arange(65132.1, 104185.7, 10000)
            level_rh = np.arange(4.2, 100, 1)
            level_tp = np.arange(0, 79.1, 1)
            proj = ccrs.Mercator(central_longitude=179.9, min_latitude=-80, max_latitude=84)

            file_temp = file_temp_nomal + "\\gfs.t00z.pgrb2.0p25.f0{}_temp.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(temp, levels=level_temp, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_temp, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_temp)
            print("success createImg temp" + str(i))

            file_pres = file_pres_nomal + "\\gfs.t00z.pgrb2.0p25.f0{}_pres.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(pres, levels=level_pres, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_pres, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_pres)
            print("success createImg pres" + str(i))

            file_rh = file_rh_nomal + "\\gfs.t00z.pgrb2.0p25.f0{}_rh.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(rh, levels=level_rh, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_rh, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_rh)
            print("success createImg rh" + str(i))

            file_tp = file_tp_nomal + "\\gfs.t00z.pgrb2.0p25.f0{}_tp.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(tp, levels=level_tp, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_tp, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_tp)
            print("success createImg tp" + str(i))
        elif i >= 100:
            # file_path = r'E:\Project\nc_data\2022-12-15\gfs.t00z.pgrb2.0p25_1hr.f008.nc'
            file_path = r'E:\Project\nc_data\{}\gfs.t00z.pgrb2.0p25.f{}.nc'.format(today, i)
            ds = xr.open_dataset(file_path)

            temp = ds['t'][:].T  # 温度：t->Temperature
            pres = ds['pres'][:].T  # 压力：pres->Pressure
            rh = ds['r2'][:].T  # 相对湿度：rh->2 metre relative humidity
            tp = ds['tp'][:].T  # 降水：tp->Total Precipitation

            level_temp = np.arange(218.1, 331.9, 1)
            level_pres = np.arange(65132.1, 104185.7, 10000)
            level_rh = np.arange(4.2, 100, 1)
            level_tp = np.arange(0, 79.1, 1)
            proj = ccrs.Mercator(central_longitude=179.9, min_latitude=-80, max_latitude=84)

            file_temp = file_temp_nomal + "\\gfs.t00z.pgrb2.0p25.f{}_temp.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(temp, levels=level_temp, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_temp, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_temp)
            print("success createImg temp" + str(i))

            file_pres = file_pres_nomal + "\\gfs.t00z.pgrb2.0p25.f{}_pres.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(pres, levels=level_pres, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_pres, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_pres)
            print("success createImg pres" + str(i))

            file_rh = file_rh_nomal + "\\gfs.t00z.pgrb2.0p25.f{}_rh.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(rh, levels=level_rh, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_rh, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_rh)
            print("success createImg rh" + str(i))

            file_tp = file_tp_nomal + "\\gfs.t00z.pgrb2.0p25.f{}_tp.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(tp, levels=level_tp, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_tp, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_tp)
            print("success createImg tp" + str(i))


# 整合了下载数据、将gfs数据转成nc和创建图像
if __name__ == '__main__':
    # 获取当天日期，以年-月-日的形式
    today = datetime.date.today()  # 2022-12-16
    today1 = str(today).replace('-', '')  # 20221216
    print("today: ", today)
    print("today1: ", today1)

    # ============1. 下载数据============
    starttime = datetime.datetime.now()
    print("starttime_download", starttime)

    file = "E:\\Project\\gfs_data\\" + str(today)
    file1 = "E:\\Project\\nc_data\\" + str(today)
    mkdir(file)
    mkdir(file1)

    # 120小时数据爬取
    file_download = "E:\\Project\\gfs_data\\{}\\".format(today)
    for i in range(121):
        downloaddata(i, file_download)
    # downloaddata(8)
    print("-----------------dowmloadData success-----------------")

    # ============2. 将gfs数据转成nc数据============
    endtime = datetime.datetime.now()
    print("endtime_download", endtime)
    print("endtime - starttime_download", endtime - starttime)

    starttime = datetime.datetime.now()
    print("starttime_change_gfs_to_nc", starttime)

    file_open_gfs = "E:\\Project\\gfs_data\\{}".format(today)
    file_change_to_nc = "E:\\Project\\nc_data\\{}".format(today)
    change_gfs_to_nc(121, file_open_gfs, file_change_to_nc)
    print("-----------------change_to_nc success-----------------")

    endtime = datetime.datetime.now()
    print("endtime_change_gfs_to_nc", endtime)
    print("endtime - starttime_change_gfs_to_nc", endtime - starttime)

    # ============3. 画出图片============
    file_temp_nomal = "E:\\Project\\figure\\{}\\temp".format(today)
    file_pres_nomal = "E:\\Project\\figure\\{}\\pres".format(today)
    file_rh_nomal = "E:\\Project\\figure\\{}\\rh".format(today)
    file_tp_nomal = "E:\\Project\\figure\\{}\\tp".format(today)
    mkdir(file_temp_nomal)
    mkdir(file_pres_nomal)
    mkdir(file_rh_nomal)
    mkdir(file_tp_nomal)

    starttime = datetime.datetime.now()
    print("starttime_createImg", starttime)

    creatimg(121, file_temp_nomal, file_pres_nomal, file_rh_nomal, file_tp_nomal)
    print("-----------------createImg success-----------------")

    endtime = datetime.datetime.now()
    print("endtime_createImg", endtime)
    print("endtime - starttime_createImg", endtime - starttime)
