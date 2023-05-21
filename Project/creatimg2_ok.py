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

    file_path = "./nc_data/2022-12-12/gfs.t00z.pgrb2.0p25_1hr.f001.nc"
    ds = xr.open_dataset(file_path)

    lat = ds.latitude
    lon = ds.longitude

    temp = (ds['Temperature'][:].T - 273.15)
    pres = ds['Pressure'][:].T
    gust = ds['Wind speed (gust)'][:].T
    ugrd = ds['U component of wind'][:].T
    ugrd10m = ds['10 metre U wind component'][:].T
    ugrd100m = ds['100 metre U wind component'][:].T
    vgrd = ds['V component of wind'][:].T
    vgrd10m = ds['10 metre V wind component'][:].T
    vgrd100m = ds['100 metre V wind component'][:].T
    rh = ds['Relative Humidity'][:].T
    rh2m = ds['2 metre relative humidity'][:].T
    t2m = ds['2 metre temperature'][:].T
    prec = ds['Precipitation rate'][:].T

    levels1 = np.arange(-120, 50, 1)
    levels2 = np.arange(40000, 120000, 10000)
    levels3 = np.arange(0, 50, 1)
    #
    levels4 = np.arange(0, 50, 1)
    levels5 = np.arange(0, 50, 1)
    levels6 = np.arange(0, 50, 1)
    levels7 = np.arange(0, 50, 1)
    levels8 = np.arange(0, 50, 1)
    levels9 = np.arange(0, 50, 1)
    levels10 = np.arange(-180, 180, 1)
    levels11 = np.arange(-180, 90, 1)
    levels12 = np.arange(-180, 180, 1)
    levels13 = np.arange(-180, 180, 1)
    proj = ccrs.Mercator(central_longitude=125.0)

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

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(gust, levels=levels3, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/wind.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(ugrd, levels=levels4, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/U.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(ugrd10m, levels=levels5, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/U10.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(ugrd100m, levels=levels6, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/U100.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(vgrd, levels=levels7, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/v.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(vgrd10m, levels=levels8, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/v10.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(vgrd100m, levels=levels9, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/v100.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(rh, levels=levels10, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/rh.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(rh2m, levels=levels11, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/rh2m.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(t2m, levels=levels12, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/t2m.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig = plt.figure(figsize=[10, 8], facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.contourf(prec, levels=levels13, cmap='Spectral_r')
    plt.axis('off')
    plt.savefig("./figure/prec.png", bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    creatimg()
