import pygrib
import numpy as np
import netCDF4 as nc


def changeto_nc():
    newfile = nc.Dataset('./nc_data/2022-12-12/gfs.t00z.pgrb2.0p25_1hr.f001.nc', 'w', format='NETCDF4')
    file_name = './gfs_data/2022-12-12/gfs.t00z.pgrb2.0p25_1hr.f001'

    ds = pygrib.open(file_name)

    long = newfile.createDimension('longitude', size=1440)
    lati = newfile.createDimension('latitude', size=721)

    lon = newfile.createVariable('lon', 'f4', dimensions='longitude')
    lat = newfile.createVariable('lat', 'f4', dimensions='latitude')

    temps = newfile.createVariable('Temperature', 'f4', dimensions=('longitude', 'latitude'))
    press = newfile.createVariable('Pressure', 'f4', dimensions=('longitude', 'latitude'))
    gusts = newfile.createVariable('Wind speed (gust)', 'f4', dimensions=('longitude', 'latitude'))
    # apcps = newfile.createVariable('precipitation', 'f4', dimensions=('longitude', 'latitude'))
    UGRD = newfile.createVariable('U component of wind', 'f4', dimensions=('longitude', 'latitude'))
    UGRD10M = newfile.createVariable('10 metre U wind component', 'f4', dimensions=('longitude', 'latitude'))
    UGRD100M = newfile.createVariable('100 metre U wind component', 'f4', dimensions=('longitude', 'latitude'))
    VGRD = newfile.createVariable('V component of wind', 'f4', dimensions=('longitude', 'latitude'))
    VGRD10M = newfile.createVariable('10 metre V wind component', 'f4', dimensions=('longitude', 'latitude'))
    VGRD100M = newfile.createVariable('100 metre V wind component', 'f4', dimensions=('longitude', 'latitude'))
    RH = newfile.createVariable('Relative Humidity', 'f4', dimensions=('longitude', 'latitude'))
    RH2M = newfile.createVariable('2 metre relative humidity', 'f4', dimensions=('longitude', 'latitude'))
    T2M = newfile.createVariable('2 metre temperature', 'f4', dimensions=('longitude', 'latitude'))
    Prec = newfile.createVariable('Precipitation rate', 'f4', dimensions=('longitude', 'latitude'))

    temp = np.rot90(np.rot90(np.rot90(ds.select(name='Temperature')[0].values)))
    pres = np.rot90(np.rot90(np.rot90(ds.select(name='Pressure')[0].values)))
    gust = np.rot90(np.rot90(np.rot90(ds.select(name='Wind speed (gust)')[0].values)))
    # apcp = np.rot90(np.rot90(np.rot90(ds.select(name='Total Precipitation')[0].values)))
    ugrd = np.rot90(np.rot90(np.rot90(ds.select(name='U component of wind')[0].values)))
    ugrd10m = np.rot90(np.rot90(np.rot90(ds.select(name='10 metre U wind component')[0].values)))
    ugrd100m = np.rot90(np.rot90(np.rot90(ds.select(name='100 metre U wind component')[0].values)))
    vgrd = np.rot90(np.rot90(np.rot90(ds.select(name='V component of wind')[0].values)))
    vgrd10m = np.rot90(np.rot90(np.rot90(ds.select(name='10 metre V wind component')[0].values)))
    vgrd100m = np.rot90(np.rot90(np.rot90(ds.select(name='100 metre V wind component')[0].values)))
    rh = np.rot90(np.rot90(np.rot90(ds.select(name='Relative humidity')[0].values)))
    rh2m = np.rot90(np.rot90(np.rot90(ds.select(name='2 metre relative humidity')[0].values)))
    t2m = np.rot90(np.rot90(np.rot90(ds.select(name='2 metre temperature')[0].values)))
    prec = np.rot90(np.rot90(np.rot90(ds.select(name='Precipitation rate')[0].values)))

    lon[:] = np.arange(0, 360, 0.25)
    lat[:] = np.arange(-90, 90.25, 0.25)

    temps[...] = temp
    press[...] = pres
    gusts[...] = gust
    # apcps[...] = apcp
    UGRD[...] = ugrd
    UGRD10M[...] = ugrd10m
    UGRD100M[...] = ugrd100m
    VGRD[...] = vgrd
    VGRD10M[...] = vgrd10m
    VGRD100M[...] = vgrd100m
    RH[...] = rh
    RH2M[...] = rh2m
    T2M[...] = t2m
    Prec[...] = prec

    lon.description = 'longitude, west is negative'
    lon.units = 'degrees east'
    lat.description = 'latitude, south is negative'
    lat.units = 'degrees north'
    temps.description = 'Temperature, random value generated by numpy'
    temps.units = 'degree'
    temps.coordinates = "lat lon"
    press.description = 'Pressure, random value generated by numpy'
    press.units = 'degree'
    press.coordinates = "lat lon"
    gusts.description = 'Wind speed (gust), random value generated by numpy'
    gusts.units = 'degree'
    gusts.coordinates = "lat lon"
    # apcps.description = 'precipitation, random value generated by numpy'
    # apcps.units = 'degree'
    # apcps.coordinates = "lat lon"
    UGRD.description = 'ugrd, random value generated by numpy'
    UGRD.units = 'degree'
    UGRD.coordinates = "lat lon"
    UGRD10M.description = 'ugrd10m, random value generated by numpy'
    UGRD10M.units = 'degree'
    UGRD10M.coordinates = "lat lon"
    UGRD100M.description = 'ugrd100m, random value generated by numpy'
    UGRD100M.units = 'degree'
    UGRD100M.coordinates = "lat lon"
    VGRD.description = 'vgrd, random value generated by numpy'
    VGRD.units = 'degree'
    VGRD.coordinates = "lat lon"
    VGRD10M.description = 'vgrd10m, random value generated by numpy'
    VGRD10M.units = 'degree'
    VGRD10M.coordinates = "lat lon"
    VGRD100M.description = 'vgrd100m, random value generated by numpy'
    VGRD100M.units = 'degree'
    VGRD100M.coordinates = "lat lon"
    RH.description = 'rh, random value generated by numpy'
    RH.units = 'degree'
    RH.coordinates = "lat lon"
    RH2M.description = 'rh2m, random value generated by numpy'
    RH2M.units = 'degree'
    RH2M.coordinates = "lat lon"
    T2M.description = 'rh2m, random value generated by numpy'
    T2M.units = 'degree'
    T2M.coordinates = "lat lon"
    Prec.description = 'prec, random value generated by numpy'
    Prec.units = 'degree'
    Prec.coordinates = "lat lon"

    newfile.close()


if __name__ == '__main__':
    changeto_nc()
