import requests


def downloaddata(time):
    try:
        if time < 10:
            res = res = requests.get(
                'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?'
                'file=gfs.t00z.pgrb2.0p25.f00{}&lev_surface=on'
                '&var_APCP=on&var_GUST=on&var_PRES=on&var_RH=on&var_TMP=on'
                '&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.20211124%2F00%2Fatmos'.format(time))
        else:
            res = requests.get(
                'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?'
                'file=gfs.t00z.pgrb2.0p25.f0{}&lev_surface=on'
                '&var_APCP=on&var_GUST=on&var_PRES=on&var_RH=on&var_TMP=on'
                '&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.20211124%2F00%2Fatmos'.format(time))

        if res.status_code == 200:
            if time < 10:
                localpath = '../Data/gfs.t00z.pgrb2.1p00.f00{}'.format(time)
            else:
                localpath = '../Data/gfs.t00z.pgrb2.0p25.f0{}'.format(time)
            open(localpath, 'wb').write(res.content)

        print("下载成功")
    except:
        print("下载失败")

downloaddata(72)
