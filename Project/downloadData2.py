import requests
import os


def downloaddata(time):
    try:
        if time < 10:
            res = requests.get(
                'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?'
                'file=gfs.t00z.pgrb2.1p00.anl&all_var=on'
                '&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.20221209%2F00%2Fatmos'.format(time))
        else:
            res = requests.get(
                'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?'
                'file=gfs.t00z.pgrb2.1p00.anl&all_var=on'
                '&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.20221209%2F00%2Fatmos'.format(time))
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?
        # file=gfs.t00z.pgrb2.1p00.anl&all_var=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.20221209%2F00%2Fatmos

        if res.status_code == 200:
            if time < 10:
                localpath = r'E:\Project\Data\气象数据\GFS\gfs.t00z.pgrb2.1p00.f00{}'.format(time)
            else:
                localpath = r'E:\Project\Data\气象数据\GFS\gfs.t00z.pgrb2.1p00.f0{}'.format(time)
            print('contents of URL written to ' + localpath)
            open(localpath, 'wb').write(res.content)
        print("下载成功")

    # # 文件保存路径
    # root_save_path = r'E:\Project\Data\气象数据\GFS'
    # # 下载数据年月日
    # year = [2022]
    # month = [12]
    # day = [9]
    # for forecast_time in range(0, time):
    #     save_path = root_save_path + '/' + str(year) + str(month).zfill(2) + str(day).zfill(2) + "_" + "00" + "_atmos/"
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
    #     date = str(year) + str(month).zfill(2) + str(day).zfill(2)
    #     file_name = "gfs.t" + "00" + "z.pgrb2.1p00.f" + str(forecast_time).zfill(3)
    #     data_file = os.path.join(save_path, file_name)
    #     if not os.path.exists(data_file):
    #         url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs." + date + "/" + "00" + "/atmos/" + file_name
    #         down_url(url, save_path + file_name)

    except:
        print("下载失败")


if __name__ == '__main__':
    downloaddata(72)

#     for year in root_year:
#         for month in root_mouth:
#             for day in root_day:
#                 for start_time in ['00']:
#                     for forecast_time in range(0, 72):
#                         save_path = root_save_path + '/' + str(year) + str(month).zfill(2) + str(day).zfill(2) + "_" + start_time + "_atmos/"
#                         if not os.path.exists(save_path):
#                             os.mkdir(save_path)
#                         date = str(year) + str(month).zfill(2) + str(day).zfill(2)
#                         file_name = "gfs.t" + start_time + "z.pgrb2.1p00.f" + str(forecast_time).zfill(3)
#                         data_file = os.path.join(save_path, file_name)
#                         if not os.path.exists(data_file):
#                             url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs." + date + "/" + start_time + "/atmos/" + file_name
#                             down_url(url, save_path + file_name)
