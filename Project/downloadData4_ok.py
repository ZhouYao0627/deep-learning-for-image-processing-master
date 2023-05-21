import requests
import datetime
import os


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  This folder is already have!  ---")


def downloaddata(time):
    try:
        if time < 10:
            res = requests.get(
                'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_low_cloud_bottom_level=on&lev_surface=on&var_APCP=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos'
                    .format(time, str(today1)))
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&all_lev=on&all_var=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&all_var=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&var_GUST=on&var_PRES=on&var_RH=on&var_TMAX=on&var_TMIN=on&var_TMP=on&var_UGRD=on&var_U-GWD=on&var_VGRD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&var_GUST=on&var_PRATE=on&var_PRES=on&var_RH=on&var_TMP=on&var_UGRD=on&var_VGRD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&var_GUST=on&var_PRES=on&var_TMP=on&var_UGRD=on&var_VGRD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_2_m_above_ground=on&var_GUST=on&var_PRES=on&var_TMP=on&var_UGRD=on&var_VGRD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&all_lev=on&var_GUST=on&var_PRES=on&var_TMP=on&var_UGRD=on&var_VGRD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_2_m_above_ground=on&all_var=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_0.1_mb=on&lev_2_m_above_ground=on&lev_max_wind=on&all_var=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_2_m_above_ground=on&var_PRES=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_10_m_above_ground=on&var_PRES=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_surface=on&var_PRES=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.anl&lev_surface=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_surface=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_surface=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_2_m_above_ground=on&lev_surface=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_10_m_above_ground=on&lev_surface=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_2_mb=on&lev_surface=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_2_hybrid_level=on&lev_surface=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_low_cloud_bottom_level=on&lev_surface=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?file=gfs.t00z.pgrb2.0p25.f00{]&lev_2_m_above_ground=on&lev_low_cloud_bottom_level=on&lev_surface=on&var_APCP=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_surface=on&var_APCP=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_2_m_above_ground=on&lev_surface=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_2_m_above_ground=on&lev_surface=on&var_APCP=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_2_m_above_ground=on&lev_low_cloud_bottom_level=on&lev_surface=on&var_APCP=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f00{}&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_low_cloud_bottom_level=on&lev_surface=on&var_APCP=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos
        elif 10 <= time < 100:
            res = requests.get(
                'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f0{}&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_low_cloud_bottom_level=on&lev_surface=on&var_APCP=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos'
                    .format(time, str(today1)))
        elif time >= 100:
            res = requests.get(
                'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t00z.pgrb2.0p25.f{}&lev_10_m_above_ground=on&lev_2_m_above_ground=on&lev_low_cloud_bottom_level=on&lev_surface=on&var_APCP=on&var_GUST=on&var_PRES=on&var_PWAT=on&var_RH=on&var_TMP=on&var_U-GWD=on&var_V-GWD=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.{}%2F00%2Fatmos'
                    .format(time, str(today1)))
        if res.status_code == 200:
            if time < 10:
                localpath = r"E:\Project\gfs_data\{}\gfs.t00z.pgrb2.0p25.f00{}".format(today, time)
            elif 10 <= time < 100:
                localpath = r'E:\Project\gfs_data\{}\gfs.t00z.pgrb2.0p25.f0{}'.format(today, time)
            elif time >= 100:
                localpath = r'E:\Project\gfs_data\{}\gfs.t00z.pgrb2.0p25.f{}'.format(today, time)
            print('contents of URL written to ' + localpath)
            open(localpath, 'wb').write(res.content)
        print("下载成功")
    except:
        print("下载失败")


if __name__ == '__main__':
    # 获取当天日期，以年-月-日的形式
    today = datetime.date.today()  # 2022-12-10
    today1 = str(today).replace('-', '')  # 20221210
    print("today: ", today)
    print("today1: ", today1)

    starttime = datetime.datetime.now()
    print(starttime)

    # 1.需要七天168小时的数据 √ 2.每天建一个文件夹 √ 3.每个文件夹中七天168小时数据 √
    file = "E:\\Project\\gfs_data\\" + str(today)
    file1 = "E:\\Project\\nc_data\\" + str(today)
    mkdir(file)
    mkdir(file1)

    # 168小时数据爬取
    for i in range(121):
        downloaddata(i)
    # downloaddata(8)
    print("dowmloadData success")

    endtime = datetime.datetime.now()
    print(endtime - starttime)
