import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib
import cartopy.crs as ccrs
from PIL import Image
import datetime
import cmaps


# u, v, 风速(gust)这三个合成一个json文件


def ROTATE(path):
    image = Image.open(path)
    out = image.transpose(Image.Transpose.ROTATE_270)  # 逆时针旋转270度
    out.save(path)


def creatimg(time):
    matplotlib.use('Agg')

    plt.rcParams.update({'font.size': 20})
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    for i in range(time):
        if 0 < i < 10:
            # file_path = r'E:\Project\nc_data\2022-12-15\gfs.t00z.pgrb2.0p25_1hr.f008.nc'
            # file_path = r'E:\Project\nc_data\{}\gfs.t00z.pgrb2.0p25.f00{}.nc'.format(today, i)
            file_path = r'E:\Project\nc_data\2023-01-09\gfs.t00z.pgrb2.0p25.f001.nc'
            ds = xr.open_dataset(file_path)

            level_temp = ds['t'][:].T - 273.15  # 温度：t->Temperature
            level_pres = ds['sp'][:].T / 100  # 压力：sp->Pressure
            level_rh = ds['r2'][:].T  # 相对湿度：rh->2 metre relative humidity
            level_tp = ds['tp'][:].T * 10  # 降水：tp->Total Precipitation

            # level_temp = np.arange(-65, 65, 1)
            # level_pres = np.arange(65132.1, 104185.7, 10000)
            # level_rh = np.arange(4.2, 100, 1)
            # level_tp = np.arange(0, 79.1, 1)
            proj = ccrs.Mercator(central_longitude=179.9, min_latitude=-80, max_latitude=84)

            diffmap_temp = ["#800000", "#8c0000", "#990000", "#ab0000", "#bf0000", "#c90000", "#d70000", "#e40000",
                            "#fa0000", "#ff1300", "#ff3700", "#ff4e00", "#ff6500", "#ff8800", "#ff9d00", "#ffb400",
                            "#ffc700", "#ffd400", "#ffee07", "#fff83e", "#ffff61", "#f5fe7f", "#e1fdb7", "#cafdf5",
                            "#c3fcff", "#bdfaff", "#b1f6ff", "#a9f3ff", "#9deeff", "#96e9ff", "#91e4ff", "#8bdfff",
                            "#83d5ff", "#80d2ff", "#78cbff", "#6fc3ff", "#4fa5ff", "#499eff", "#3b88ff", "#2e71ff",
                            "#2b68ff", "#2152fd", "#1e4bfc", "#173af3", "#102ae8", "#0917db", "#0207d1"]
            diffmap_rh = ["#384774", "#386FA6", "#3885AE", "#388EAE", "#389EAE", "#36AF94", "#69AE38", "#AE9238",
                          "#AE6E38", "#AE6E38", "#AE6E38", "#AE6E38"]
            # diffmap_pres_normal = ["#006794", "#006794", "#006794", "#006794", "#007593", "#128693", "#489A98",
            #                        "#8CB2A8", "#B2B09D", "#A78EC7", "#A36F3F", "#A0522C", "#A0522C"]
            # diffmap_pres_noraml2 = ["#A0522C", "#A0522C", "#A36F3F", "#A78EC7", "#B2B09D", "#8CB2A8", "#489A98", "#128693",
            #                 "#007593", "#006794", "#006794", "#006794", "#006794"]
            # diffmap_pres = ["#006794", "#006794", "#006794", "#006794", "#007593", "#128693", "#489A98", "#8CB2A8",
            #                 "#B2B09D", "#A78EC7", "#A36F3F", "#A0522C", "#A0522C"]
            diffmap_pres = ["#A0522C", "#A0522C", "#A36F3F", "#A78EC7", "#B2B09D", "#8CB2A8", "#489A98", "#128693",
                            "#007593", "#006794", "#006794", "#006794", "#006794"]
            # diffmap_tp_normal = ["#615882", "#615882", "#615882", "#615882", "#49668E", "#34758F", "#2A7B8C", "#1F8189",
            #               "#0B8D82", "#5C9A64", "#FB9EBF", "#F9A2C1", "#F9A2C1"]
            # diffmap_tp_normal1 = ["#F9A2C1", "#F9A2C1", "#FB9EBF", "#5C9A64", "#0B8D82", "#1F8189", "#2A7B8C", "#34758F",
            #               "#49668E", "#615882", "#615882", "#615882", "#615882"]
            # diffmap_tp_normal2 = ["#F9A2C1", "#FB9EBF", "#5C9A64", "#0B8D82", "#1F8189", "#2A7B8C", "#34758F",
            #                       "#49668E", "#615882"]
            diffmap_tp = ["#F9A2C1", "#F9A2C1", "#FB9EBF", "#5C9A64", "#0B8D82", "#1F8189", "#2A7B8C", "#34758F",
                          "#49668E", "#615882", "#615882", "#615882", "#615882"]
            levers_temp = [-60, -43, -41, -39, -37, -35, -33, -31, -29, -27, -25, -23, -21, -19, -17, -15, -13, -11, -9,
                           -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
                           41, 43, 60]
            levers_rh = [0, 10, 20, 30, 45, 50, 55, 75, 85, 92, 97, 100]
            levers_pres = [489, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1065]
            # levers_tp = [0, 5, 10, 20, 30, 40, 100, 200, 300] # 原
            levers_tp = [0, 0.2, 0.4, 0.8, 1, 1.5, 1.8, 2, 3.73, 4, 200]

            lat = ds["latitude"].values
            lon = ds["longitude"].values

            level_temp1 = np.flip(level_temp, axis=1)
            level_rh1 = np.flip(level_rh, axis=1)
            level_pres1 = np.flip(level_pres, axis=1)
            level_tp1 = np.flip(level_tp, axis=1)

            file_temp = "E:\\Project\\figure\\temp\\temp{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(lat, lon, level_temp1, levers_temp, colors=diffmap_temp[::-1])
            plt.axis('off')
            plt.savefig(file_temp, bbox_inches='tight', pad_inches=0, dpi=600)
            plt.close()
            ROTATE(file_temp)
            print("success temp" + str(i))

            file_pres = "E:\\Project\\figure\\pres\\pres{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(lat, lon, level_pres1, levers_pres, colors=diffmap_pres[::-1])
            plt.axis('off')
            plt.savefig(file_pres, bbox_inches='tight', pad_inches=0, dpi=600)
            plt.close()
            ROTATE(file_pres)
            print("success pres" + str(i))

            file_rh = "E:\\Project\\figure\\rh\\rh{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(lat, lon, level_rh1, levers_rh, colors=diffmap_rh[::-1])
            plt.axis('off')
            plt.savefig(file_rh, bbox_inches='tight', pad_inches=0, dpi=600)
            plt.close()
            ROTATE(file_rh)
            print("success rh" + str(i))

            file_tp = "E:\\Project\\figure\\tp\\tp{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(lat, lon, level_tp1, levers_tp, colors=diffmap_tp[::-1])
            plt.axis('off')
            plt.savefig(file_tp, bbox_inches='tight', pad_inches=0, dpi=600)
            plt.close()
            ROTATE(file_tp)
            print("success tp" + str(i))
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

            file_temp = "E:\\Project\\figure\\temp\\temp{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(temp, levels=level_temp, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_temp, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_temp)
            print("success temp" + str(i))

            file_pres = "E:\\Project\\figure\\pres\\pres{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(pres, levels=level_pres, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_pres, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_pres)
            print("success pres" + str(i))

            file_rh = "E:\\Project\\figure\\rh\\rh{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(rh, levels=level_rh, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_rh, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_rh)
            print("success rh" + str(i))

            file_tp = "E:\\Project\\figure\\tp\\tp{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(tp, levels=level_tp, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_tp, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_tp)
            print("success tp" + str(i))
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

            file_temp = "E:\\Project\\figure\\temp\\temp{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(temp, levels=level_temp, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_temp, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_temp)
            print("success temp" + str(i))

            file_pres = "E:\\Project\\figure\\pres\\pres{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(pres, levels=level_pres, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_pres, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_pres)
            print("success pres" + str(i))

            file_rh = "E:\\Project\\figure\\rh\\rh{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(rh, levels=level_rh, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_rh, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_rh)
            print("success rh" + str(i))

            file_tp = "E:\\Project\\figure\\tp\\tp{}.png".format(i)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection=proj)
            ax.contourf(tp, levels=level_tp, cmap='Spectral_r')
            plt.axis('off')
            plt.savefig(file_tp, bbox_inches='tight', pad_inches=0)
            plt.close()
            ROTATE(file_tp)
            print("success tp" + str(i))


if __name__ == '__main__':
    today = datetime.date.today()  # 2023-01-05
    print(today)

    starttime = datetime.datetime.now()
    print(starttime)

    creatimg(2)
    print("createImg success")

    endtime = datetime.datetime.now()
    print(endtime - starttime)
