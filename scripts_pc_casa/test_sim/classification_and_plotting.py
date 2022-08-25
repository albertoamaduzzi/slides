#HCA analysis for curve shapes
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
import scipy.spatial.distance as ssd
from scipy.signal import savgol_filter
import scipy.cluster.hierarchy as sch
import time
import branca.colormap as cm
import geopy
import geopy.distance
import folium
from folium.plugins import BeautifyIcon
import dtw
import os
import argparse
import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides','script_pc_casa','test_sim'))
    from simulator_script import *
    from analyzer_script import *
    from sim_objects import *
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e

def norm(a):
  tot = a.sum()
  if tot!=0: return a/tot
  else:      return a

def dist(a,b,t='euclidean'):
  if t == 'euclidean':
    return np.sqrt(np.sum((a-b)**2)) # or   ssd.euclidean(a,b)
  elif t == 'correlation':
    return ssd.correlation(a,b)
  elif t == 'dtw':
    return dtw.dtw(a,b,distance_only=True).distance
    # with radius 8 and 15min data interval there is a 2 hour range for dtw

def inv_barriers(s):
  lc = s[-1]
  if lc == '0': return s[:-1]+'1'
  else:         return s[:-1]+'0'




class classifier:
    def __init__(self,sim):
        self.dist_type = 'euclidean'
        self.hca_method = 'ward'
        self.nan_t = 5 # %. droppa le barriers con troppi nan
        self.sm_window = 9 # smoothing window, deve essere dispari
        self.sm_order = 3 # smoothing polynomial order
        self.use_smooth = True # usare i dati smooth o i dati raw
        self.normalize = True # normalizzare i dati per giornata
        self.date_format = '%Y-%m-%d %H:%M:%S'

        self.std_correlation_matrix = initialize_standard_correlation_matrix(sim)









class plotter:
    def __init__(self,sim):

