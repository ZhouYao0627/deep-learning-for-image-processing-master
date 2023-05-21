import os
import requests
import time
import calendar


def save_data(url, FILENAME):
    try:
        result = requests.get(url, stream=True, timeout=30)
        result.raise_for_status()
        buff = b""
        t1 = time.time()
        all_size = int(result.headers['Content-Length'])
        for chunk in result.iter_content(1000 * 1024):
            buff += chunk
            t2 = time.time()
            print(url, len(buff), all_size, len(buff) / all_size, len(buff) / (t2 - t1), t2 - t1, (t2 - t1) / len(buff) * (all_size - len(buff)))
        f = open(FILENAME, 'wb')
        f.write(buff)
        f.close()
        print('contents of URL written to ' + FILENAME)
        return True
    except:
        print('requests请求错误,重连中... ')
        return False


def down_url(url, FILENAME):
    open_state = save_data(url, FILENAME)
    while not open_state:
        open_state = save_data(url, FILENAME)


def run_it():
    # 文件保存路径
    root_save_path = r'E:\Project\Data\气象数据\GFS'
    # 下载数据年月日
    root_year = [2022]
    root_mouth = [12]
    root_day = [9]
    for year in root_year:
        for month in root_mouth:
            for day in root_day:
                for start_time in ['00']:
                    for forecast_time in range(0, 72):
                        save_path = root_save_path + '/' + str(year) + str(month).zfill(2) + str(day).zfill(2) + "_" + start_time + "_atmos/"
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        date = str(year) + str(month).zfill(2) + str(day).zfill(2)
                        file_name = "gfs.t" + start_time + "z.pgrb2.1p00.f" + str(forecast_time).zfill(3)
                        data_file = os.path.join(save_path, file_name)
                        if not os.path.exists(data_file):
                            url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs." + date + "/" + start_time + "/atmos/" + file_name
                            down_url(url, save_path + file_name)
# https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?file=gfs.t00z.pgrb2.1p00.anl&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.20221209%2F00%2Fatmos
if __name__ == '__main__':
    run_it()
