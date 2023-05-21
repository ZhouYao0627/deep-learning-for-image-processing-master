import xarray as xr
import cfgrib
import netCDF4 as nc
import datetime


# data = xr.open_dataset(r'E:\Project\gfs_data\2022-12-15\gfs.t00z.pgrb2.0p25_1hr.f008', engine='cfgrib')
# data.to_netcdf(r'E:\Project\nc_data\2022-12-15\gfs.t00z.pgrb2.0p25_1hr.f008.nc')

# path = r'E:\Project\nc_data\2022-12-14\gfs.t00z.pgrb2.0p25_1hr.f001.nc'
# data = xr.open_dataset(path)

# print("data: ", data)


def change_gfs_to_nc(time):
    for i in range(time):
        if i < 10:
            data = xr.open_dataset(r'E:\Project\gfs_data\{}\gfs.t00z.pgrb2.0p25.f00{}'.format(today, i),
                                   engine='cfgrib')
            data.to_netcdf(r'E:\Project\nc_data\{}\gfs.t00z.pgrb2.0p25.f00{}.nc'.format(today, i))
        elif 10 <= i < 100:
            data = xr.open_dataset(r'E:\Project\gfs_data\{}\gfs.t00z.pgrb2.0p25.f0{}'.format(today, i), engine='cfgrib')
            data.to_netcdf(r'E:\Project\nc_data\{}\gfs.t00z.pgrb2.0p25.f0{}.nc'.format(today, i))
        elif i >= 100:
            data = xr.open_dataset(r'E:\Project\gfs_data\{}\gfs.t00z.pgrb2.0p25.f{}'.format(today, i), engine='cfgrib')
            data.to_netcdf(r'E:\Project\nc_data\{}\gfs.t00z.pgrb2.0p25.f{}.nc'.format(today, i))
        print('success: ' + str(i))


if __name__ == '__main__':
    today = datetime.date.today()  # 2022-12-15
    print(today)

    starttime = datetime.datetime.now()
    print(starttime)

    change_gfs_to_nc(121)
    print("change_to_nc success")

    endtime = datetime.datetime.now()
    print(endtime - starttime)
