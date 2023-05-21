import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

from map_funcs import add_Chinese_provinces, set_map_extent_and_ticks

if __name__ == '__main__':
    # 设置绘图区域.
    lonmin, lonmax = 75, 150
    latmin, latmax = 15, 60
    extents = [lonmin, lonmax, latmin, latmax]

    # 读取extents区域内的数据.
    filename = 't_uv_rh_gp_ERA5.nc'
    with xr.open_dataset(filename) as ds:
        # ERA5文件的纬度单调递减,所以先反转过来.
        ds = ds.sortby(ds.latitude)
        ds = ds.isel(time=0).sel(
            longitude=slice(lonmin, lonmax),
            latitude=slice(latmin, latmax),
            level=500
        )

    proj = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)

    # 添加海岸线和中国省界.
    ax.coastlines(resolution='10m', lw=0.3)
    add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
    # 设置经纬度刻度.
    set_map_extent_and_ticks(
        ax, extents,
        xticks=np.arange(-180, 190, 15),
        yticks=np.arange(-90, 100, 15),
        nx=1, ny=1
    )
    ax.tick_params(labelsize='small')

    # 画出相对湿度的填色图.
    im = ax.contourf(
        ds.longitude, ds.latitude, ds.r,
        levels=np.linspace(0, 100, 11), cmap='RdYlBu_r',
        extend='both', alpha=0.8
    )
    cbar = fig.colorbar(
        im, ax=ax, shrink=0.9, pad=0.1, orientation='horizontal',
        format=mticker.PercentFormatter()
    )
    cbar.ax.tick_params(labelsize='small')

    # 画出风箭头.直接使用DataArray会报错,所以转换成ndarray.
    Q = ax.quiver(
        ds.longitude.values, ds.latitude.values,
        ds.u.values, ds.v.values,
        scale_units='inches', scale=180, angles='uv',
        units='inches', width=0.008, headwidth=4,
        regrid_shape=20, transform=proj
    )
    # 在ax右下角腾出放图例的空间.
    # zorder需大于1,以避免被之前画过的内容遮挡.
    w, h = 0.12, 0.12
    rect = mpatches.Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes,
        fc='white', ec='k', lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
    # 添加风箭头的图例.
    qk = ax.quiverkey(
        Q, X=1-w/2, Y=0.7*h, U=40,
        label=f'{40} m/s', labelpos='S', labelsep=0.05,
        fontproperties={'size': 'x-small'}
    )

    title = 'Relative Humidity and Wind at 500 hPa'
    ax.set_title(title, fontsize='medium')

    fig.savefig('rh_wnd.png', dpi=200, bbox_inches='tight')
    plt.close(fig)