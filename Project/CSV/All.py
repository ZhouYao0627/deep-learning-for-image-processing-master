import numpy as np
import netCDF4 as nc
import pandas as pd
import datetime
import csv
import os
import shutil
import matplotlib.pyplot as plt
import cv2
import requests
import xarray as xr
import matplotlib as mpl


def rgb2hex(r, g, b):
    """将 RGB 颜色转换为十六进制颜色"""
    hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
    return hex_color


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print(path + ": ---  new folder...  ---")
        print("---  OK  ---")
    else:
        print(path + ":---  This folder is already have!  ---")


def convertCSV(day, param):
    letters = []  # 新建空列表
    for char in range(ord('a'), ord('x') + 1):  # ord() 函数返回字符的 ASCII 码值
        letters.append(chr(char))  # chr() 函数返回指定 ASCII 码值对应的字符
    if day == 1:
        Day1(day, letters, param)
    elif day == 2:
        Day2(day, letters, param)
    elif day == 3:
        Day3(day, letters, param)
    elif day == 4:
        Day4(day, letters, param)
    elif day == 5:
        Day5(day, letters, param)


def Day1(day, letters, param):
    for i in range(0, 24):
        if i <= 8:
            ncfile_read = "E:\\Project\\QQFY\\" + str(today) + "\\dataNC\\gfs.t00z.pgrb2.0p25.f00" + str(i + 1) + ".nc"
            ncfile = nc.Dataset(ncfile_read)
            ncFileToCsv(day, letters[i], ncfile, param)
        else:
            ncfile_read = "E:\\Project\\QQFY\\" + str(today) + "\\dataNC\\gfs.t00z.pgrb2.0p25.f0" + str(i + 1) + ".nc"
            ncfile = nc.Dataset(ncfile_read)
            ncFileToCsv(day, letters[i], ncfile, param)


def Day2(day, letters, param):
    j = 0
    ncfile_read = "E:\\Project\\QQFY\\" + str(today) + "\\dataNC\\gfs.t00z.pgrb2.0p25.f0"
    for i in range(24, 48):
        ncfile = nc.Dataset(ncfile_read + str(i + 1) + '.nc')
        ncFileToCsv(day, letters[j], ncfile, param)
        j = j + 1


def Day3(day, letters, param):
    j = 0
    ncfile_read = "E:\\Project\\QQFY\\" + str(today) + "\\dataNC\\gfs.t00z.pgrb2.0p25.f0"
    for i in range(48, 72):
        ncfile = nc.Dataset(ncfile_read + str(i + 1) + '.nc')
        ncFileToCsv(day, letters[j], ncfile, param)
        j = j + 1


def Day4(day, letters, param):
    j = 0
    ncfile_read = "E:\\Project\\QQFY\\" + str(today) + "\\dataNC\\gfs.t00z.pgrb2.0p25.f0"
    for i in range(72, 96):
        ncfile = nc.Dataset(ncfile_read + str(i + 1) + '.nc')
        ncFileToCsv(day, letters[j], ncfile, param)
        j = j + 1


def Day5(day, letters, param):
    j = 0
    for i in range(96, 120):
        if i <= 98:
            ncfile_read = "E:\\Project\\QQFY\\" + str(today) + "\\dataNC\\gfs.t00z.pgrb2.0p25.f0"
            ncfile = nc.Dataset(ncfile_read + str(i + 1) + '.nc')
            ncFileToCsv(day, letters[j], ncfile, param)
            j = j + 1
        else:
            ncfile_read = "E:\\Project\\QQFY\\" + str(today) + "\\dataNC\\gfs.t00z.pgrb2.0p25.f"
            ncfile = nc.Dataset(ncfile_read + str(i + 1) + '.nc')
            ncFileToCsv(day, letters[j], ncfile, param)
            j = j + 1


def ncFileToCsv(day, letters, ncfile, param):
    data = ncfile.variables[param][:]
    # 转换数据形状以避免字符串中包含多个值
    reshaped_data = data.reshape((-1, data.shape[-1]))
    rounded_data = np.round(reshaped_data, decimals=1)  # 四舍五入保留1位小数
    # 将数据存储为 CSV 文件
    df = pd.DataFrame(rounded_data)
    if param == "r2":
        file_r2_CSV = file_r2CSV + str(day) + letters + ".csv"
        df.to_csv(file_r2_CSV, index=False, header=None)
    elif param == "t2m":
        file_t2m_CSV = file_t2mCSV + str(day) + letters + ".csv"
        df.to_csv(file_t2m_CSV, index=False, header=None)
    elif param == "sp":
        file_sp_CSV = file_spCSV + str(day) + letters + ".csv"
        df.to_csv(file_sp_CSV, index=False, header=None)
    elif param == "tp":
        file_tp_CSV = file_tpCSV + str(day) + letters + ".csv"
        df.to_csv(file_tp_CSV, index=False, header=None)


def chazhi(day, name):
    letters = []  # 新建空列表
    for char in range(ord('a'), ord('x') + 1):  # ord() 函数返回字符的 ASCII 码值
        letters.append(chr(char))  # chr() 函数返回指定 ASCII 码值对应的字符
    if day == 1:
        readAndSave(day, letters, name)
    elif day == 2:
        readAndSave(day, letters, name)
    elif day == 3:
        readAndSave(day, letters, name)
    elif day == 4:
        readAndSave(day, letters, name)
    elif day == 5:
        readAndSave(day, letters, name)


def readAndSave(day, letters, name):
    if name == "r2CSV":
        file_name = "E:\\Project\\QQFY\\" + str(today) + "\\r2\\" + name + "\\"
        saveNew(day, letters, file_name)
    elif name == "t2mCSV":
        file_name = "E:\\Project\\QQFY\\" + str(today) + "\\t2m\\" + name + "\\"
        saveNew(day, letters, file_name)
    elif name == "spCSV":
        file_name = "E:\\Project\\QQFY\\" + str(today) + "\\sp\\" + name + "\\"
        saveNew(day, letters, file_name)
    elif name == "tpCSV":
        file_name = "E:\\Project\\QQFY\\" + str(today) + "\\tp\\" + name + "\\"
        saveNew(day, letters, file_name)


def saveNew(day, letters, file_name):
    for k in range(len(letters) - 1):
        file_name_1 = file_name + str(day) + letters[k] + '.csv'
        with open(file_name_1, 'r') as file1:
            reader1 = csv.reader(file1)
            data1 = list(reader1)

        # 读取第二个csv文件
        file_name_2 = file_name + str(day) + letters[k + 1] + '.csv'
        with open(file_name_2, 'r') as file2:
            reader2 = csv.reader(file2)
            data2 = list(reader2)

        data_middle = [[None for _ in range(1440)] for _ in range(721)]
        data_csv = [[None for _ in range(1440)] for _ in range(721)]

        # 获取数据的行数和列数
        rows = len(data1)
        cols = len(data1[0])

        # 对相同位置的数据进行相加运算
        for i in range(rows):
            for j in range(cols):
                data_middle[i][j] = (float(data2[i][j]) - float(data1[i][j])) / 2
                data_csv[i][j] = (float(data1[i][j]) + float(data_middle[i][j]))

        # 将结果写入新的csv文件
        with open(file_name + str(day) + letters[k] + letters[k] + '.csv', 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(data_csv)


def rename_files(folder_path):
    # 获取文件列表，并按照顺序排序
    files = os.listdir(folder_path)
    sorted_files = sorted(files)

    # 遍历文件并重命名
    for i, f in enumerate(sorted_files):
        # 构造新的文件名
        new_name = "{index}{ext}".format(
            index=i + 1,
            ext=os.path.splitext(f)[1]  # 获取文件扩展名
        )
        old_path = os.path.join(folder_path, f)
        new_path = os.path.join(folder_path, new_name)
        shutil.move(old_path, new_path)  # 重命名文件
        print("File {0} has been renamed to {1}.".format(f, new_name))


def draw1(name):
    if name == "r2CSV":
        cmap = 'rainbow_r'
        file_r2_save = "E:\\Project\\QQFY\\" + str(today) + "\\r2\\r2Figure\\"
        draw2(file_r2_save, cmap, "")
    elif name == "t2mCSV":
        cmap = 'rainbow'
        file_t2m_save = "E:\\Project\\QQFY\\" + str(today) + "\\t2m\\t2mFigure\\"
        draw2(file_t2m_save, cmap, "")
    elif name == "spCSV":
        mycolors = [rgb2hex(0, 72, 255), rgb2hex(0, 102, 255), rgb2hex(0, 153, 255), rgb2hex(0, 204, 255),
                    rgb2hex(0, 255, 234), rgb2hex(0, 255, 183), rgb2hex(0, 255, 115), rgb2hex(0, 255, 34),
                    rgb2hex(128, 255, 0), rgb2hex(166, 255, 0), rgb2hex(200, 255, 0), rgb2hex(247, 255, 0),
                    rgb2hex(255, 221, 0), rgb2hex(255, 191, 0), rgb2hex(255, 170, 0), rgb2hex(255, 119, 0),
                    rgb2hex(255, 89, 0), rgb2hex(255, 50, 0), rgb2hex(255, 30, 0)]
        bounds = [70000, 80000, 90000, 95000, 97000, 98000, 99000, 100000, 100500, 101000, 101100, 101200, 101300,
                  101400, 101500, 101700, 101800, 102000, 102500, 102600]
        cmap = mpl.colors.ListedColormap(mycolors)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        file_sp_save = "E:\\Project\\QQFY\\" + str(today) + "\\sp\\spFigure\\"
        draw2(file_sp_save, cmap, norm)
    elif name == "tpCSV":
        mycolors = [rgb2hex(255, 255, 255), rgb2hex(0, 174, 255), rgb2hex(0, 144, 255), rgb2hex(0, 102, 255),
                    rgb2hex(0, 57, 255)]
        bounds = [0, 0.1, 0.11, 0.25, 0.26, 0.7]
        cmap = mpl.colors.ListedColormap(mycolors)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        file_tp_save = "E:\\Project\\QQFY\\" + str(today) + "\\tp\\tpFigure\\"
        draw2(file_tp_save, cmap, norm)


def draw2(file, cmap, norm):
    if file == "E:\\Project\\QQFY\\" + str(today) + "\\r2\\r2Figure\\":
        file_read_r2 = file_r2CSV
        draw3(file, cmap, file_read_r2)
    elif file == "E:\\Project\\QQFY\\" + str(today) + "\\t2m\\t2mFigure\\":
        file_read_t2m = file_t2mCSV
        draw3(file, cmap, file_read_t2m)
    elif file == "E:\\Project\\QQFY\\" + str(today) + "\\sp\\spFigure\\":
        file_read_sp = file_spCSV
        draw4(file, cmap, norm, file_read_sp)
    elif file == "E:\\Project\\QQFY\\" + str(today) + "\\tp\\tpFigure\\":
        file_read_tp = file_tpCSV
        draw4(file, cmap, norm, file_read_tp)


def draw3(file, cmap, fileRead):
    for i in range(235):
        file_read = fileRead + str(i + 1) + ".csv"
        df = pd.read_csv(file_read)
        data = df.values
        data = np.reshape(data, (720, 1440))
        plt.imshow(data, cmap=cmap, interpolation='nearest')
        plt.axis('off')
        file_save = file + str(i + 1) + ".png"
        plt.savefig(file_save, dpi=200, bbox_inches='tight', pad_inches=0)


def draw4(file, cmap, norm, fileRead):
    for i in range(235):
        file_read = fileRead + str(i + 1) + ".csv"
        df = pd.read_csv(file_read)
        data = df.values
        data = np.reshape(data, (720, 1440))
        plt.imshow(data, cmap=cmap, norm=norm, interpolation='nearest')
        plt.axis('off')
        file_save = file + str(i + 1) + ".png"
        plt.savefig(file_save, dpi=200, bbox_inches='tight', pad_inches=0)


def movie1(name):
    # file_r2_movie = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[0] + "\\" + dir_name1[0] + dir_name2[2] + "\\"
    # file_t2m_movie = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[1] + "\\" + dir_name1[1] + dir_name2[2] + "\\"
    # file_tp_movie = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[2] + "\\" + dir_name1[2] + dir_name2[2] + "\\"
    # file_sp_movie = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[3] + "\\" + dir_name1[3] + dir_name2[2] + "\\"
    # D:\\software\\IDEA\\Project\\RWQX\\src\\main\\resources\\video\\fy
    file_movie_save = "D:\\software\\IDEA\\Project\\RWQX\\src\\main\\resources\\video\\fy\\"
    mkdir(file_movie_save)

    if name == "r2":
        file_r2_figure = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[0] + "\\" + dir_name1[0] + dir_name2[
            1] + "\\"
        movie2(file_r2_figure, file_movie_save, name)
    elif name == "t2m":
        file_t2m_figure = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[1] + "\\" + dir_name1[1] + dir_name2[
            1] + "\\"
        movie2(file_t2m_figure, file_movie_save, name)
    elif name == "tp":
        file_tp_figure = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[2] + "\\" + dir_name1[2] + dir_name2[
            1] + "\\"
        movie2(file_tp_figure, file_movie_save, name)
    elif name == "sp":
        file_sp_figure = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[3] + "\\" + dir_name1[3] + dir_name2[
            1] + "\\"
        movie2(file_sp_figure, file_movie_save, name)


def movie2(file_picture, file_movie, name):
    img_array = []
    for i in range(1, 236):
        filename = file_picture + str(i) + '.png'
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    file_movie_name = file_movie + name + '.mp4'
    out = cv2.VideoWriter(file_movie_name, cv2.VideoWriter_fourcc(*'DIVX'), 12, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def downloaddata(time, file_download):
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
                localpath = file_download + "gfs.t00z.pgrb2.0p25.f00{}".format(time)
            elif 10 <= time < 100:
                localpath = file_download + 'gfs.t00z.pgrb2.0p25.f0{}'.format(time)
            elif time >= 100:
                localpath = file_download + 'gfs.t00z.pgrb2.0p25.f{}'.format(time)
            print('contents of URL written to ' + localpath)
            open(localpath, 'wb').write(res.content)
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


"""
    全球风云总体步骤：
    0. 爬取gfs数据
    1. gfs数据转换为nc文件
    2. nc文件转换为CSV文件
    3. 插值
    4. 修改名称
    5. 绘画
    6. 制成视频
"""
if __name__ == '__main__':
    day = 5  # 需要设置的天数
    today = datetime.date.today()  # 2023-05-21
    today1 = str(today).replace('-', '')  # 20230521

    dir_name1 = ["r2", "t2m", "tp", "sp"]
    dir_name2 = ["CSV", "Figure", 'Movie']
    dir_name3 = ["dataGFS", "dataNC"]
    dir_name4 = ["r2CSV", "t2mCSV", "tpCSV", "spCSV"]  # 因为上面已经创建了该文件夹

    for name1 in range(len(dir_name1)):
        for name2 in range(len(dir_name2)):
            file = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[name1] + "\\" + dir_name1[name1] + dir_name2[
                name2] + "\\"
            mkdir(file)
    for name3 in range(len(dir_name3)):
        file = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name3[name3] + "\\"
        mkdir(file)

    file_r2CSV = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[0] + "\\" + dir_name4[0] + "\\"
    file_t2mCSV = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[1] + "\\" + dir_name4[1] + "\\"
    file_tpCSV = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[2] + "\\" + dir_name4[2] + "\\"
    file_spCSV = "E:\\Project\\QQFY\\" + str(today) + "\\" + dir_name1[3] + "\\" + dir_name4[3] + "\\"
    # ========== 0. 爬取gfs数据 ==========
    file_download = "E:\\Project\\QQFY\\{}\\dataGFS\\".format(today)
    for i in range(121):
        downloaddata((i + 1), file_download)
    # ========== 1. gfs数据转换为nc文件 ==========
    file_open_gfs = file_download
    file_change_to_nc = "E:\\Project\\QQFY\\{}\\dataNC\\".format(today)
    change_gfs_to_nc(121, file_open_gfs, file_change_to_nc)
    # ========== 2. nc文件转换为CSV文件 ==========
    for j in range(len(dir_name1)):
        for i in range(day):
            convertCSV((i + 1), dir_name1[j])
    # ========== 3. 插值 ==========
    for name4 in range(len(dir_name4)):
        for i in range(day):
            chazhi(i + 1, dir_name4[name4])
    # ========== 4. 修改名称 ==========
    rename_files(file_r2CSV)
    rename_files(file_t2mCSV)
    rename_files(file_tpCSV)
    rename_files(file_spCSV)
    # ========== 5. 绘画 ==========
    draw1(dir_name4[0])  # 湿度
    draw1(dir_name4[1])  # 温度
    draw1(dir_name4[2])  # 降水
    draw1(dir_name4[3])  # 压强
    # ========== 6. 制成视频 ==========
    for i in range(len(dir_name1)):
        movie1(dir_name1[i])
