import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib
import cartopy.crs as ccrs


def creatimg():
    matplotlib.use('Agg')

    plt.rcParams.update({'font.size': 20})
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    file_path = "./nc_data/2022-12-12/gfs.t00z.pgrb2.0p25_1hr.f002.nc"
    ds = xr.open_dataset(file_path)

    lat = ds.latitude
    lon = ds.longitude

    temp = (ds['temperature'][:].T - 273.15)
    pres = ds['pressure'][:].T
    # prec = ds['precipitation'][:].T
    wind = ds['wind'][:].T

    levels1 = np.arange(-60, 65, 1)
    levels2 = np.arange(40000, 120000, 10000)
    # levels3 = np.arange(0, 50, 1)
    levels4 = np.arange(0, 50, 1)
    proj = ccrs.Mercator(central_longitude=125.0)
    # proj = ccrs.WGS84_SEMIMAJOR_AXIS(central_longitude=0)

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(temp, levels=levels1, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/temp.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(pres, levels=levels2, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/pres.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # fig = plt.figure(figsize=[10, 8], facecolor='none')
    # ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    # ax.contourf(prec, levels=levels3, cmap='Spectral_r')
    # plt.axis('off')
    # plt.savefig("../../statics/qqfy/images/0{}/prec.png", bbox_inches='tight', pad_inches=0)
    # plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(wind, levels=levels4, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/wind.png", bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    creatimg()
