import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.mpl.ticker as mticker
import cartopy.crs as ccrs
import cartopy.mpl.ticker as mticker

nc_file = "E:\\Project\\Data\\2020-12-1.nc"

xr_data = xr.open_dataset(nc_file, engine="netcdf4")
for b in range(1):
    t2m = xr_data["t2m"].values.__array__()[b]
    print(xr_data["t2m"].values.__array__()[b])
    for i in range(721):
        for j in range(1440):
            t2m[i][j] = t2m[i][j]-273
    lat = xr_data["latitude"].values
    lon = xr_data["longitude"].values
    # save_file = "D:\\all\\sp\\" + str(b) +".jpg"
    save_file = "D:\\Software\\Pycharm\\Project\\Figure\\t2m" + str(b) +".jpg"
    proj = ccrs.PlateCarree(central_longitude=180)

    fig = plt.figure(figsize=(9, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    cs = ax.contourf(lon, lat, t2m, transform=proj, cmap='RdBu_r')  # RdBu_r nipy_spectral
    # ax.quiver(lon, lat, u, v, transform=ccrs.PlateCarree(), regrid_shape=35, width=0.002)
    ax.coastlines(color='dimgray')
    ax.set_global()

    cbar = fig.colorbar(cs, orientation='vertical', pad=0.02, aspect=20, shrink=0.6)
    cbar.set_label('Pha')

    # xticks = [-180, -120, -60, 0, 60, 120, 180]
    # ax.set_xticks(xticks, crs=proj)
    # ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=proj)
    lon_formatter = mticker.LongitudeFormatter(zero_direction_label=True)
    lat_formatter = mticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    plt.savefig(save_file, dpi=300)
    plt.show()

