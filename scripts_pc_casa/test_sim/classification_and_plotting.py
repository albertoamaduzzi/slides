#HCA analysis for curve shapes
import sys
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
#import dtw
import os
import argparse
import datetime
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

#try:
sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides','script_pc_casa','test_sim'))
from simulator_script import *
from analyzer_script import *
from sim_objects import *
#except Exception as e:
#  raise Exception('library loading error : {}'.format(e)) from e

def norm(a):
  tot = a.sum()
  if tot!=0: return a/tot
  else:      return a

def dist(a,b,t='euclidean'):
  if t == 'euclidean':
    return np.sqrt(np.sum((a-b)**2)) # or   ssd.euclidean(a,b)
  elif t == 'correlation':
    return ssd.correlation(a,b)
#  elif t == 'dtw':
#    return dtw.dtw(a,b,distance_only=True).distance
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
    '''plotter.__dict__:
    CONSTANT OF THE ANALYSIS:
    coilsd: pandas dataframe -> columns [Description;ID;Lat;Lon], rows BARRIER_NAME_(1,2,3,..)_(IN,OUT)
    topn_distance: int -> number of couples of barriers whose distance is smallest 
    topn_corr: int -> number of couples of top barriers whose correlation is biggest
    center_coords: list -> [] center of barriers in coordinates
    zoom: starting zoom for each map
    '''
    def __init__(self,sim,analysis):
      print('plotter')
      self.topn_distance = 7
      self.topn_corr = 7
      self.zoom = 10
      self.list_keys_dict_df_ward = ['df_dist_sim','df_dist_sim_norm','df_dist_real','df_dist_real_norm']
      self.windows = analysis.windows
      self.coilsdf = pd.read_csv(sim.simcfgorig['file_barrier'], sep=';') 
      self.center_coords = self.coilsdf[['Lat', 'Lon']].mean().values
      self.tags = sim.tags
      self.tagn = sim.tagn
      self.str0 = analysis.str0
# setup
    def common_map(self,sim,ch):
      '''Returns mappa with coords:
      TODO: add sources and attractions and color them if they are active sources and active attractions
      sim.df_avg columns: [NAME_BARRIER_IN_OUT,timestamp]:rows -> counts
      for s in sim.dict_sources:
          if s.is_added:
              
              
              
          if s.is_changed:'''
      ## ESSENTIAL VARIABLES ##
      print('common map')
      coord_mat = {}
      if self.windows:
        if sim.average_fluxes:
          with open(sim.state_basename + '/dict_euclidean_distances_avg_{}_sim.json'.format(self.str0),'r') as outfile:
            self.dict_df_ward = json.load(outfile)
        else:
          with open(sim.state_basename + '/dict_euclidean_distances_{}_sim.json'.format(self.str0),'r') as outfile:
            self.dict_df_ward = json.load(outfile)
      else:
        if sim.average_fluxes:
          with open(sim.state_basename + '\\dict_euclidean_distances_normed_avg_{}_sim.json'.format(self.str0),'r') as outfile:
            self.dict_df_ward = json.load(outfile)
        else:
          with open(sim.state_basename + '\\dict_euclidean_distances_normed_{}_sim.json'.format(self.str0),'r') as outfile:
            self.dict_df_ward = json.load(outfile)
      ## START MAPPA ##
      mappa = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)
      for s in self.tags:
        coords = list(self.coilsdf[self.coilsdf['Description']==s][['Lat','Lon']].values[0])
        coord_mat[s] = coords
      maxv, minv = np.max(self.dict_df_ward[self.list_keys_dict_df_ward[0]]['distance'].values), np.min(self.dict_df_ward[self.list_keys_dict_df_ward[0]].values)
      colormap = cm.LinearColormap(colors=['blue','yellow','red'], index=[minv,maxv/2,maxv], vmin=minv, vmax=maxv)
      for s in self.tags:
          if s in list(ch.dict_sources.keys()):
            if not ch.dict_sources[s].is_reset:
              color = 'red'
          elif s in list(ch.dict_attractions.keys()):       
            if not ch.dict_attractions[s].is_reset:
              color = 'green'
          else:
              color = 'blue'
      folium.Marker(coord_mat[s]['Lat'],coord_mat[s]['Lon'], popup=s, icon = folium.Icon(color=color)).add_to(mappa)
      n_epoch = 0
      if os.path.exists(os.path.join(sim.state_basename,'map_{}'.format(n_epoch))):
          os.mkdir(os.path.join(sim.state_basename,'map_{}'.format(n_epoch)))
          mappa.save(os.path.join(sim.state_basename,'map_{}'.format(n_epoch),'barriers_all.html'))

      '''    # COLORING THE POPUPS
      for s in tags:
        for source_ in list(ch.dict_sources.keys()):
          if s ch.dict_sources[source_]
        
        for attraction_ in list(ch.dict_attractions.keys()):
          ch.dict_attractions[attraction_]
      # top N
      for key in self.list_keys_dict_df_ward:
      dataslice = data[:topn].copy()
      mappa = folium.Map(location=center_coords, tiles='cartodbpositron', control_scale=True, zoom_start=9)
      for cid, row in dataslice.iterrows():
        inp,out = row['barrier1'], row['barrier2']
        folium.Marker(location=coord_mat[inp], popup=inp).add_to(mappa)
        folium.Marker(location=coord_mat[out], popup=out).add_to(mappa)
        folium.PolyLine([coord_mat[inp],coord_mat[out]],color=colormap(row[ordtype]),weight=2,popup='{}>{}:{}'.format(inp,out,row[ordtype])).add_to(mappa)

      mappa.add_child(colormap)
      mappa.save(os.path.join(saving_dir,f'barriers_top_{topn}.html'))
      
      mappa = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)
      return mappa
      '''              

