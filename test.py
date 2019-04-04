import os
from os import listdir
from os.path import isfile, join
import yaml

from caldb_scaler_pointlike import CalDB

# Load configuration file
config = yaml.load(open('myconfig.cfg', 'r'))

# 'DB' Input directory
caldb = config['general']['caldb']
os.environ['CALDB'] = caldb

# IRF repository of interest
irf_indir = config['general']['irf_indir']

# List all files in directory
file_list = [f for f in listdir(join(caldb, irf_indir)) if isfile(join(caldb, irf_indir, f))]

# Only on file for test
filetest = file_list[0]

# Initialise caldb
caldb = CalDB('irf_south', filetest, verbose=True)

# Scale IRFs
caldb.scale_irf(config)


