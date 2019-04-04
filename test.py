import os
import argparse
import yaml

from caldb_scaler_pointlike import CalDB

config = yaml.load(open('myconfig.cfg', 'r'))

os.environ['CALDB'] = '/Users/julien/Documents/WorkingDir/Tools/python/gpropa_paper/irf/'

caldb = CalDB('irf_south', 'CTA-Performance-South-20deg-S-05h_20170627.fits.gz', verbose=True)
caldb.scale_irf(config)


