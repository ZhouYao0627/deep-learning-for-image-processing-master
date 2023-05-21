import requests
import os


def downloaddata(time):
    try:
        res = requests.get(
            'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?'
            'file=gfs.t00z.pgrb2.1p00.anl&all_var=on'
            '&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.20221209%2F00%2Fatmos'.format(time))
        # https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?
        # file=gfs.t00z.pgrb2.1p00.anl&all_var=on
        # &leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.20221209%2F00%2Fatmos

        if res.status_code == 200:
            localpath = r'E:\Project\Data\气象数据\GFS\gfs.t00z.pgrb2.1p00.f0{}'.format(time)
            open(localpath, 'wb').write(res.content)
        print("下载成功")
    except:
        print("下载失败")


if __name__ == '__main__':
    downloaddata(72)
