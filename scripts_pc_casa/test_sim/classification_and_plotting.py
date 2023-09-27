#HCA analysis for curve shapes
import sys
import pandas as pd
import numpy as np
import matplotlib.colors as colors
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
import scipy.spatial.distance as ssd
from scipy.signal import savgol_filter
import scipy.cluster.hierarchy as sch
import time
import branca.colormap as cm
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
    def __init__(self,sim,n_epoch):
        self.dist_type = 'euclidean'
        self.hca_method = 'ward'
        self.nan_t = 5 # %. droppa le barriers con troppi nan
        self.sm_window = 9 # smoothing window, deve essere dispari
        self.sm_order = 3 # smoothing polynomial order
        self.use_smooth = True # usare i dati smooth o i dati raw
        self.normalize = True # normalizzare i dati per giornata
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.n_epoch = n_epoch
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
    def __init__(self,sim,analysis,n_epoch):
      print('plotter')
      self.topn_distance = 7
      self.topn_corr = 7
      self.zoom = 15
      self.list_keys_dict_df_ward = ['df_dist_sim','df_dist_sim_norm','df_dist_real','df_dist_real_norm']
      self.windows = analysis.windows
      self.coilsdf = pd.read_csv(sim.simcfgorig['file_barrier'], sep=';') 
      self.center_coords = self.coilsdf[['Lat', 'Lon']].mean().values
      self.tags = sim.tags
      self.tagn = sim.tagn
      self.str0 = analysis.str0
      self.dict_df_ward = analysis.dict_df_ward
      self.n_epoch = n_epoch
      self.topn = 8
      
    def add_popup_mappa_circle(self,mappa,s,ch,coord_mat):
      '''Adds a popup to the map with name s and coords coord_mar[s]'''
      try:
        name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
        if name == 'Scalzi_3_IN':
            color = 'blue'
            popup = folium.Popup(s,parse_html = True)            
            folium.Marker(location = coord_mat[s], popup= popup, icon = folium.Icon(color=color)).add_to(mappa)
            folium.Circle(location = coord_mat[s], popup= popup, radius = 1000, weight = 2, icon = folium.Icon(color=color)).add_to(mappa)                  
        elif s in [x.upper() for x in list(ch.dict_sources.keys())]:
          name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
          if not ch.dict_sources[name].is_reset or ch.dict_sources[name].is_added or ch.dict_sources[name].is_changed or ch.dict_sources[name].is_default:
            color = 'blue'
            popup = folium.Popup(s,parse_html = True)            
            folium.Marker(location = coord_mat[s], popup= popup, icon = folium.Icon(color=color)).add_to(mappa)
            folium.Circle(location = coord_mat[s], popup= popup, radius = 1000, weight = 1, icon = folium.Icon(color=color)).add_to(mappa)    
        elif s in [x.upper() for x in list(ch.dict_attractions.keys())]:       
          name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
          if not ch.dict_attractions[name].is_reset or ch.dict_attractions[name].is_added or  ch.dict_attractions[name].is_changed or ch.dict_attractions[name].is_default:
            color = 'blue'
            popup = folium.Popup(s,parse_html = True)
            folium.Marker(location = coord_mat[s], popup= popup, icon = folium.Icon(color=color)).add_to(mappa)
        else:
            color = 'blue'
            popup = folium.Popup(s,parse_html = True)
            folium.Marker(location = coord_mat[s], popup= popup, icon = folium.Icon(color=color)).add_to(mappa)
        return mappa
      except KeyError:
        print('cannot add ',s)
        return mappa
      
    def common_map_in_circle(self,sim,ch,map_):
      '''Plots mappa_in and mappa_out with coords IN and OUT format hrml folium:
      Colors: green if attraction, red if source, blue simple barriers'''
      ## ESSENTIAL VARIABLES ##
      print('common map in')
      coord_mat = {} 
      ## START MAPPA ##      
      mappa_in = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)
      for s in self.tags:
        try:
          coords = list(self.coilsdf[self.coilsdf['Description']==s][['Lat','Lon']].values[0])
          coord_mat[s] = coords
          self.coord_mat = coord_mat
        except IndexError:
          print('the barrier {} is not present in the barriers'.format(s))
#      print(coord_mat)
      for s in self.tags:
        if not '_OUT' in s:
          mappa_in = self.add_popup_mappa_circle(mappa_in,s,ch,coord_mat)
        else:
          pass          
      for attraction in list(ch.dict_attractions.keys()):
        popup = folium.Popup(attraction,parse_html = True)
        folium.Marker(location = (ch.dict_attractions[attraction].lat,ch.dict_attractions[attraction].lon), popup= popup, icon = folium.Icon(color='green')).add_to(mappa_in)
      if self.windows:
        if self.n_epoch == 0:
          map_ = '/map_{}/'.format(0)
        else:
          map_ = map_ + '/map_{}/'.format(self.n_epoch)
      else:
        if self.n_epoch == 0:
          map_ = '\\map_{}\\'.format(0)
        else:
          map_ = map_ + '\\map_{}\\'.format(self.n_epoch)
      if not os.path.exists(sim.state_basename + map_):
          os.mkdir(sim.state_basename + map_)
      mappa_in.save(sim.state_basename + map_ +'barriers_all_in_circle.html')
      return map_




      

      
# setup
    def add_popup_mappa(self,mappa,s,ch,coord_mat):
      '''Adds a popup to the map with name s and coords coord_mar[s]'''
      print('map in ',s)
      try:
        name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
        if s in [x.upper() for x in list(ch.dict_sources.keys())]:
          name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
          if not ch.dict_sources[name].is_reset or ch.dict_sources[name].is_added or ch.dict_sources[name].is_changed or ch.dict_sources[name].is_default or ch.dict_sources[name].name == 'Schiavoni_1_IN':
            color = 'red'
        elif s in [x.upper() for x in list(ch.dict_attractions.keys())]:       
          name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
          if not ch.dict_attractions[name].is_reset or ch.dict_attractions[name].is_added or  ch.dict_attractions[name].is_changed or ch.dict_attractions[name].is_default:
            color = 'green'
        else:
            color = 'blue'
        popup = folium.Popup(s,parse_html = True)

        folium.Marker(location = coord_mat[s], popup= popup, icon = folium.Icon(color=color)).add_to(mappa)
        return mappa
      except KeyError:
        print('cannot add ',s)
        return mappa
      

    def common_map_in(self,sim,ch,map_):
      '''Plots mappa_in and mappa_out with coords IN and OUT format hrml folium:
      Colors: green if attraction, red if source, blue simple barriers'''
      ## ESSENTIAL VARIABLES ##
      print('common map in')
      coord_mat = {} 
      ## START MAPPA ##      
      mappa_in = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)
      for s in self.tags:
        try:
          coords = list(self.coilsdf[self.coilsdf['Description']==s][['Lat','Lon']].values[0])
          coord_mat[s] = coords
          self.coord_mat = coord_mat
        except IndexError:
          print('the barrier {} is not present in the barriers'.format(s))
#      print(coord_mat)
      for s in self.tags:
        if not '_OUT' in s:
          mappa_in = self.add_popup_mappa(mappa_in,s,ch,coord_mat)
          popup = folium.Popup('Schiavoni_1_IN',parse_html = True)
          folium.Marker(location = [45.433819,12.344299], popup= popup, icon = folium.Icon(color='red')).add_to(mappa_in)
        else:
          pass          
      for attraction in list(ch.dict_attractions.keys()):
        popup = folium.Popup(attraction,parse_html = True)
        folium.Marker(location = (ch.dict_attractions[attraction].lat,ch.dict_attractions[attraction].lon), popup= popup, icon = folium.Icon(color='green')).add_to(mappa_in)
      if self.windows:
        if self.n_epoch == 0:
          map_ = '/map_{}/'.format(0)
        else:
          map_ = map_ + '/map_{}/'.format(self.n_epoch)
      else:
        if self.n_epoch == 0:
          map_ = '\\map_{}\\'.format(0)
        else:
          map_ = map_ + '\\map_{}\\'.format(self.n_epoch)
      if not os.path.exists(sim.state_basename + map_):
          os.mkdir(sim.state_basename + map_)
      mappa_in.save(sim.state_basename + map_ +'barriers_all_in.html')
      return map_

    def common_map_out(self,sim,ch,map_):
      '''Plots mappa_out and mappa_out with coords OUT format html folium:
      Colors: green if attraction, red if source, blue simple barriers'''
      ## ESSENTIAL VARIABLES ##
      print('common map out')
      coord_mat = {} 
      ## START MAPPA ##      
      mappa_out = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)

      for s in self.tags:
        try:
          coords = list(self.coilsdf[self.coilsdf['Description']==s][['Lat','Lon']].values[0])
          coord_mat[s] = coords
          self.coord_mat = coord_mat
        except IndexError:
          print('the barrier {} is not present in the barriers'.format(s))
#      print(coord_mat)
      for s in self.tags:
        if '_OUT' in s:
          mappa_out = self.add_popup_mappa(mappa_out,s,ch,coord_mat)
        else:
          pass          
      for attraction in list(ch.dict_attractions.keys()):
        popup = folium.Popup(attraction,parse_html = True)
        folium.Marker(location = (ch.dict_attractions[attraction].lat,ch.dict_attractions[attraction].lon), popup= popup, icon = folium.Icon(color='green')).add_to(mappa_out)
      mappa_out.save(sim.state_basename + map_ +'barriers_all_out.html')
      return map_
    def map_good_correlated_in(self,analysis,sim,ch,map_):
      print('mapping correlated IN')
      coord_mat = {} 
      ## START MAPPA ##      
      mappa_out = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)
      for s in analysis.list_good_corr:
        try:
          coords = list(self.coilsdf[self.coilsdf['Description']==s][['Lat','Lon']].values[0])
          coord_mat[s] = coords
          self.coord_mat = coord_mat
        except IndexError:
          print('the barrier {} is not present in the barriers'.format(s))
#      print(coord_mat)
      for s in analysis.list_good_corr:
        if '_IN' in s:
          mappa_out = self.add_popup_mappa(mappa_out,s,ch,coord_mat)
        else:
          pass          
      for attraction in list(ch.dict_attractions.keys()):
        popup = folium.Popup(attraction,parse_html = True)
        folium.Marker(location = (ch.dict_attractions[attraction].lat,ch.dict_attractions[attraction].lon), popup= popup, icon = folium.Icon(color='green')).add_to(mappa_out)
      mappa_out.save(sim.state_basename + map_ +'good_correlated_in.html')

    def map_good_correlated_out(self,analysis,sim,ch,map_):
      print('mapping correlated')
      coord_mat = {} 
      ## START MAPPA ##      
      mappa_out = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)
      for s in analysis.list_good_corr:
        try:
          coords = list(self.coilsdf[self.coilsdf['Description']==s][['Lat','Lon']].values[0])
          coord_mat[s] = coords
          self.coord_mat = coord_mat
        except IndexError:
          print('the barrier {} is not present in the barriers'.format(s))
#      print(coord_mat)
      print('barriers with good corr good sim \n',analysis.list_good_corr_good_sim)
      for s in analysis.list_good_corr:
        if '_OUT' in s:
          mappa_out = self.add_popup_mappa(mappa_out,s,ch,coord_mat)
        else:
          pass          
      for attraction in list(ch.dict_attractions.keys()):
        popup = folium.Popup(attraction,parse_html = True)
        folium.Marker(location = (ch.dict_attractions[attraction].lat,ch.dict_attractions[attraction].lon), popup= popup, icon = folium.Icon(color='green')).add_to(mappa_out)
      mappa_out.save(sim.state_basename + map_ +'good_correlated_out.html')

    def map_with_plots_in(self,sim,ch,map_,analysis): 
      import altair as alt
      import folium
      import pandas as pd
      import branca
      print('map with plots in')
      coord_mat = {} 
      ## START MAPPA ##      
      mappa = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)
      for s in self.tags:
        try:
          coords = list(self.coilsdf[self.coilsdf['Description']==s][['Lat','Lon']].values[0])
          coord_mat[s] = coords
        except IndexError:
          print('the barrier {} is not present in the barriers'.format(s))
      count_barriers = 0
      for s in self.tags:
        if not '_OUT' in s:
          name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
          if s in [x.upper() for x in list(ch.dict_sources.keys())]:
            name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
            if not ch.dict_sources[name].is_reset or ch.dict_sources[name].is_added or ch.dict_sources[name].is_changed or ch.dict_sources[name].is_default:
              color = 'red'
          elif s in [x.upper() for x in list(ch.dict_attractions.keys())]:       
            name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
            if not ch.dict_attractions[name].is_reset or ch.dict_attractions[name].is_added or  ch.dict_attractions[name].is_changed or ch.dict_attractions[name].is_default:
              color = 'green'
          else:
              color = 'blue'
          ##### STARTING TO PREPARE THE FOIUM GRAPHS #########
          two_charts_template = """
          <!DOCTYPE html>
          <html>
          <head>
            <script src="https://cdn.jsdelivr.net/npm/vega@{vega_version}"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-lite@{vegalite_version}"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-embed@{vegaembed_version}"></script>
          </head>
          <body>

          <div id="vis1"></div>
          <div id="vis2"></div>

          <script type="text/javascript">
            vegaEmbed('#vis1', {spec1}).catch(console.error);
            vegaEmbed('#vis2', {spec2}).catch(console.error);
          </script>
          </body>
          </html>
          """      
          if sim.pick_day:
            ######## START HTML FILE FOR DAY #############
            d = {'sim': [],'real':[],'time':[]}
            df = pd.DataFrame(d)
            try:
              df['sim'] = analysis.sim_dataframe[s][2:]
              df['real_'] = ['real data' for x in range(len(analysis.sim_dataframe[s][2:]))]
              df['sim_'] = ['simulation' for x in range(len(analysis.sim_dataframe[s][2:]))]
              df['real'] = sim.df_day[s][:-2]
              df['time'] = list(np.arange(22))
              c_sim = alt.Chart(df).mark_line().encode(x='time:N', y='sim:Q',color = 'sim_:N')
              c_real = alt.Chart(df).mark_line().encode(x='time:N', y='real:Q',color = 'real_:N').properties(title = s)#.properties(width=180,height=180).facet(column = '{}:N'.format(s)).configure_header(titleFontSize=40, labelFontSize=40)
              chart1 = c_sim + c_real
            except KeyError:
              print(s,'is not present in either df_day or sim_dataframe')
            d1 = {'difference correlation squared':[],'barrier':[]}
            df1 = pd.DataFrame(d1)
            try:
              df1['difference correlation squared'] = analysis.dif_cor[s]
              df1['barrier']= analysis.dif_cor.columns
            except KeyError:
              print(s,'is not present in analysis.dif_cor')
            chart2 = alt.Chart(df1).mark_line().encode(x='barrier:N', y='difference correlation squared:Q')
            if self.windows:
              if not os.path.exists(sim.state_basename + map_+ 'daily'):
                os.mkdir(sim.state_basename + map_+ 'daily')
              with open(sim.state_basename + map_+ 'daily/' + 'comparison_sim_real_{}.html', 'w') as f:
                f.write(two_charts_template.format(
                vega_version=alt.VEGA_VERSION,
                vegalite_version=alt.VEGALITE_VERSION,
                vegaembed_version=alt.VEGAEMBED_VERSION,
                spec1=chart1.to_json(indent=None),
                spec2=chart2.to_json(indent=None),
            ))
              html_file = open(sim.state_basename + map_+ 'daily/' + 'comparison_sim_real_{}.html', 'r', encoding='utf-8')
            else:
              if not os.path.exists(sim.state_basename + map_+ 'daily'):
                os.mkdir(sim.state_basename + map_+ 'daily')                    
              with open(sim.state_basename + map_+ 'daily\\' + 'comparison_sim_real_{}.html', 'w') as f:
                f.write(two_charts_template.format(
                vega_version=alt.VEGA_VERSION,
                vegalite_version=alt.VEGALITE_VERSION,
                vegaembed_version=alt.VEGAEMBED_VERSION,
                spec1=chart1.to_json(indent=None),
                spec2=chart2.to_json(indent=None),
            ))
              html_file = open(sim.state_basename + map_+ 'daily\\' + 'comparison_sim_real_{}.html', 'r', encoding='utf-8')
            ######## END HTML FILE FOR DAY #############                
              
          elif sim.average_fluxes:
            ######## START HTML FILE FOR AVG #############
            d = {'sim':[],'real':[],'time':[]}
            df = pd.DataFrame(d)
            try:
              df['sim'] = analysis.sim_dataframe[s][2:]
              df['real_'] = ['real data' for x in range(len(analysis.sim_dataframe[s][2:]))]
              df['sim_'] = ['simulation' for x in range(len(analysis.sim_dataframe[s][2:]))]
              df['real'] = sim.df_avg[s][:-2]
              df['time'] = list(np.arange(22))
              c_sim = alt.Chart(df).mark_line().encode(x='time:N', y='sim:Q',color = 'sim_:N')
              c_real = alt.Chart(df).mark_line().encode(x='time:N', y='real:Q',color = 'real_:N').properties(title = s)#.facet(column = '{}:N'.format(s)).configure_header(titleFontSize=40, labelFontSize=40)
              chart1 = c_sim + c_real
            except KeyError:
              print(s,'is not present in either df_avg or sim_dataframe')
            d1 = {'difference correlation squared':[],'barrier':[]}
            df1 = pd.DataFrame(d1)
            try:
              df1['difference correlation squared'] = analysis.dif_cor[s]
              df1['barrier']= analysis.dif_cor.columns
            except KeyError:
              print(s,'is not present in analysis.dif_cor')
              print('keys analysis.dif_cor:\t',analysis.dif_cor.columns)
            chart2 = alt.Chart(df1).mark_line().encode(x='barrier:N', y='difference correlation squared:Q')
            if self.windows:
              if not os.path.exists(sim.state_basename + map_+ 'averaged'):
                os.mkdir(sim.state_basename + map_+ 'averaged')
              with open(sim.state_basename + map_+ 'averaged/' +'comparison_sim_real_{}.html'.format(s), 'w') as f:
                f.write(two_charts_template.format(
                vega_version=alt.VEGA_VERSION,
                vegalite_version=alt.VEGALITE_VERSION,
                vegaembed_version=alt.VEGAEMBED_VERSION,
                spec1=chart1.to_json(indent=None),
                spec2=chart2.to_json(indent=None),
            ))
                
              html_file = open(sim.state_basename + map_+ 'averaged/' +'comparison_sim_real_{}.html'.format(s), 'r', encoding='utf-8')

            else:            
              if not os.path.exists(sim.state_basename + map_+ 'averaged'):
                os.mkdir(sim.state_basename + map_+ 'averaged')
              with open(sim.state_basename + map_+ 'averaged\\' +'comparison_sim_real_{}.html'.format(s), 'w') as f:
                f.write(two_charts_template.format(
                vega_version=alt.VEGA_VERSION,
                vegalite_version=alt.VEGALITE_VERSION,
                vegaembed_version=alt.VEGAEMBED_VERSION,
                spec1=chart1.to_json(indent=None),
                spec2=chart2.to_json(indent=None),
            ))
              html_file = open(sim.state_basename + map_+ 'averaged\\' +'comparison_sim_real_{}.html'.format(s), 'r', encoding='utf-8')
          ######## END HTML FILE FOR AVG #############
          charts_code = html_file.read() 
          try:
            iframe = branca.element.IFrame(html=charts_code, width=1500, height=400)
            popup = folium.Popup(iframe, max_width=2000)
            folium.Marker(location = coord_mat[s], popup=popup, icon = folium.Icon(color = color)).add_to(mappa)
          except KeyError:
            print('cannot add ',s)
      if self.windows:              
        mappa.save(sim.state_basename + map_ +'barriers_correlation_distance_plots_in.html')
      else:
        mappa.save(sim.state_basename + map_ +'barriers_correlation_distance_plots_in.html')
        return True

    def map_with_plots_out(self,sim,ch,map_,analysis): 
      import altair as alt
      import folium
      import pandas as pd
      import branca
      print('map with plots out')
      coord_mat = {} 
      ## START MAPPA ##      
      mappa = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)
      for s in self.tags:
        try:
          coords = list(self.coilsdf[self.coilsdf['Description']==s][['Lat','Lon']].values[0])
          coord_mat[s] = coords
        except IndexError:
          print('the barrier {} is not present in the barriers'.format(s))
      for s in self.tags:
        if '_OUT' in s:
          name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
          if s in [x.upper() for x in list(ch.dict_sources.keys())]:
            name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
            if not ch.dict_sources[name].is_reset or ch.dict_sources[name].is_added or ch.dict_sources[name].is_changed or ch.dict_sources[name].is_default:
              color = 'red'
          elif s in [x.upper() for x in list(ch.dict_attractions.keys())]:       
            name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
            if not ch.dict_attractions[name].is_reset or ch.dict_attractions[name].is_added or  ch.dict_attractions[name].is_changed or ch.dict_attractions[name].is_default:
              color = 'green'
          else:
              color = 'blue'
          ##### STARTING TO PREPARE THE FOIUM GRAPHS #########
          two_charts_template = """
          <!DOCTYPE html>
          <html>
          <head>
            <script src="https://cdn.jsdelivr.net/npm/vega@{vega_version}"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-lite@{vegalite_version}"></script>
            <script src="https://cdn.jsdelivr.net/npm/vega-embed@{vegaembed_version}"></script>
          </head>
          <body>

          <div id="vis1"></div>
          <div id="vis2"></div>

          <script type="text/javascript">
            vegaEmbed('#vis1', {spec1}).catch(console.error);
            vegaEmbed('#vis2', {spec2}).catch(console.error);
          </script>
          </body>
          </html>
          """      
          if sim.pick_day:
            ######## START HTML FILE FOR DAY #############
            d = {'sim': [],'real':[],'time':[]}
            df = pd.DataFrame(d)
            try:
              df['sim'] = analysis.sim_dataframe[s][2:]
              df['real_'] = ['real data' for x in range(len(analysis.sim_dataframe[s][2:]))]
              df['sim_'] = ['simulation' for x in range(len(analysis.sim_dataframe[s][2:]))]
              df['real'] = sim.df_day[s][:-2]
              df['time'] = list(np.arange(22))
              c_sim = alt.Chart(df).mark_line().encode(x='time:N', y='sim:Q',color = 'sim_:N')
              c_real = alt.Chart(df).mark_line().encode(x='time:N', y='real:Q',color = 'real_:N').properties(title = s)#.properties(width=180,height=180).facet(column = '{}:N'.format(s)).configure_header(titleFontSize=40, labelFontSize=40)
              chart1 = c_sim + c_real
            except KeyError:
              print(s,'is not present in either df_day or sim_dataframe')
            d1 = {'difference correlation squared':[],'barrier':[]}
            df1 = pd.DataFrame(d1)
            try:
              df1['difference correlation squared'] = analysis.dif_cor[s]
              df1['barrier']= analysis.dif_cor.columns
            except KeyError:
              print(s,'is not present in analysis.dif_cor')
            chart2 = alt.Chart(df1).mark_line().encode(x='barrier:N', y='difference correlation squared:Q')
            if self.windows:
              if not os.path.exists(sim.state_basename + map_+ 'daily'):
                os.mkdir(sim.state_basename + map_+ 'daily')
              with open(sim.state_basename + map_+ 'daily/' + 'comparison_sim_real_{}.html', 'w') as f:
                f.write(two_charts_template.format(
                vega_version=alt.VEGA_VERSION,
                vegalite_version=alt.VEGALITE_VERSION,
                vegaembed_version=alt.VEGAEMBED_VERSION,
                spec1=chart1.to_json(indent=None),
                spec2=chart2.to_json(indent=None),
            ))
              html_file = open(sim.state_basename + map_+ 'daily/' + 'comparison_sim_real_{}.html', 'r', encoding='utf-8')
            else:
              if not os.path.exists(sim.state_basename + map_+ 'daily'):
                os.mkdir(sim.state_basename + map_+ 'daily')                    
              with open(sim.state_basename + map_+ 'daily\\' + 'comparison_sim_real_{}.html', 'w') as f:
                f.write(two_charts_template.format(
                vega_version=alt.VEGA_VERSION,
                vegalite_version=alt.VEGALITE_VERSION,
                vegaembed_version=alt.VEGAEMBED_VERSION,
                spec1=chart1.to_json(indent=None),
                spec2=chart2.to_json(indent=None),
            ))
              html_file = open(sim.state_basename + map_+ 'daily\\' + 'comparison_sim_real_{}.html', 'r', encoding='utf-8')
            ######## END HTML FILE FOR DAY #############                
              
          elif sim.average_fluxes:
            ######## START HTML FILE FOR AVG #############
            d = {'sim':[],'real':[],'time':[]}
            df = pd.DataFrame(d)
            try:
              df['sim'] = analysis.sim_dataframe[s][2:]
              df['real_'] = ['real data' for x in range(len(analysis.sim_dataframe[s][2:]))]
              df['sim_'] = ['simulation' for x in range(len(analysis.sim_dataframe[s][2:]))]
              df['real'] = sim.df_avg[s][:-2]
              df['time'] = list(np.arange(22))
              c_sim = alt.Chart(df).mark_line().encode(x='time:N', y='sim:Q',color = 'sim_:N')
              c_real = alt.Chart(df).mark_line().encode(x='time:N', y='real:Q',color = 'real_:N').properties(title = s)#.facet(column = '{}:N'.format(s)).configure_header(titleFontSize=40, labelFontSize=40)
              chart1 = c_sim + c_real
            except KeyError:
              print(s,'is not present in either df_avg or sim_dataframe')
            d1 = {'difference correlation squared':[],'barrier':[]}
            df1 = pd.DataFrame(d1)
            try:
              df1['difference correlation squared'] = analysis.dif_cor[s]
              df1['barrier']= analysis.dif_cor.columns
            except KeyError:
              print(s,'is not present in analysis.dif_cor')
              print('keys analysis.dif_cor:\t',analysis.dif_cor.columns)
            chart2 = alt.Chart(df1).mark_line().encode(x='barrier:N', y='difference correlation squared:Q')
            if self.windows:
              if not os.path.exists(sim.state_basename + map_+ 'averaged'):
                os.mkdir(sim.state_basename + map_+ 'averaged')
              with open(sim.state_basename + map_+ 'averaged/' +'comparison_sim_real_{}.html'.format(s), 'w') as f:
                f.write(two_charts_template.format(
                vega_version=alt.VEGA_VERSION,
                vegalite_version=alt.VEGALITE_VERSION,
                vegaembed_version=alt.VEGAEMBED_VERSION,
                spec1=chart1.to_json(indent=None),
                spec2=chart2.to_json(indent=None),
            ))
                
              html_file = open(sim.state_basename + map_+ 'averaged/' +'comparison_sim_real_{}.html'.format(s), 'r', encoding='utf-8')

            else:            
              if not os.path.exists(sim.state_basename + map_+ 'averaged'):
                os.mkdir(sim.state_basename + map_+ 'averaged')
              with open(sim.state_basename + map_+ 'averaged\\' +'comparison_sim_real_{}.html'.format(s), 'w') as f:
                f.write(two_charts_template.format(
                vega_version=alt.VEGA_VERSION,
                vegalite_version=alt.VEGALITE_VERSION,
                vegaembed_version=alt.VEGAEMBED_VERSION,
                spec1=chart1.to_json(indent=None),
                spec2=chart2.to_json(indent=None),
            ))
              html_file = open(sim.state_basename + map_+ 'averaged\\' +'comparison_sim_real_{}.html'.format(s), 'r', encoding='utf-8')
          ######## END HTML FILE FOR AVG #############
          charts_code = html_file.read() 
          try:
            iframe = branca.element.IFrame(html=charts_code, width=1500, height=400)
            popup = folium.Popup(iframe, max_width=2000)
            folium.Marker(location = coord_mat[s], popup=popup, icon = folium.Icon(color = color)).add_to(mappa)
          except KeyError:
            print('cannot add',s)
      if self.windows:              
        mappa.save(sim.state_basename + map_ +'barriers_correlation_distance_plots_out.html')
      else:
        mappa.save(sim.state_basename + map_ +'barriers_correlation_distance_plots_out.html')
        return True

    
    def map_with_plots(self,sim,ch,map_,analysis):
      import altair as alt
      import folium
      import pandas as pd
      import branca
      print('map with plots')
      coord_mat = {} 
      ## START MAPPA ##      
      mappa = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)
      for s in self.tags:
        try:
          coords = list(self.coilsdf[self.coilsdf['Description']==s][['Lat','Lon']].values[0])
          coord_mat[s] = coords
        except IndexError:
          print('the barrier {} is not present in the barriers'.format(s))
#      print(coord_mat)
      maxv, minv = np.max(self.dict_df_ward[self.list_keys_dict_df_ward[0]]['distance']), np.min(self.dict_df_ward[self.list_keys_dict_df_ward[0]]['distance'])
      count_barriers = 0
      for s in self.tags:
        name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
        if s in [x.upper() for x in list(ch.dict_sources.keys())]:
          name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
          if not ch.dict_sources[name].is_reset or ch.dict_sources[name].is_added or ch.dict_sources[name].is_changed or ch.dict_sources[name].is_default:
            color = 'red'
        elif s in [x.upper() for x in list(ch.dict_attractions.keys())]:       
          name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
          if not ch.dict_attractions[name].is_reset or ch.dict_attractions[name].is_added or  ch.dict_attractions[name].is_changed or ch.dict_attractions[name].is_default:
            color = 'green'
        else:
            color = 'blue'
        try:
          # I use this as if  self.tags[count_barriers - 1] is not in the keys of coord_mat then I can add the coord_mat[s]           
          if coord_mat[self.tags[count_barriers - 1]] == True:
              print('the coord before')
        except KeyError:
          print(self.tags[count_barriers - 1],'is not present in coord mat, then I try with ',self.tags[count_barriers])
          try:
            two_charts_template = """
            <!DOCTYPE html>
            <html>
            <head>
              <script src="https://cdn.jsdelivr.net/npm/vega@{vega_version}"></script>
              <script src="https://cdn.jsdelivr.net/npm/vega-lite@{vegalite_version}"></script>
              <script src="https://cdn.jsdelivr.net/npm/vega-embed@{vegaembed_version}"></script>
            </head>
            <body>

            <div id="vis1"></div>
            <div id="vis2"></div>

            <script type="text/javascript">
              vegaEmbed('#vis1', {spec1}).catch(console.error);
              vegaEmbed('#vis2', {spec2}).catch(console.error);
            </script>
            </body>
            </html>
            """      
            if sim.pick_day:
              d = {'sim': [],'real':[],'time':[]}
              df = pd.DataFrame(d)
              try:
                df['sim'] = analysis.sim_dataframe[s][2:]
                df['real_'] = ['real data' for x in range(len(analysis.sim_dataframe[s][2:]))]
                df['sim_'] = ['simulation' for x in range(len(analysis.sim_dataframe[s][2:]))]
                df['real'] = sim.df_day[s][:-2]
                df['time'] = list(np.arange(22))
                c_sim = alt.Chart(df).mark_line().encode(x='time:N', y='sim:Q',color = 'sim_:N')
                c_real = alt.Chart(df).mark_line().encode(x='time:N', y='real:Q',color = 'real_:N').properties(title = s)#.properties(width=180,height=180).facet(column = '{}:N'.format(s)).configure_header(titleFontSize=40, labelFontSize=40)
                chart1 = c_sim + c_real
#                text = chart1.mark_text(align='center').encode(text='{}:N'.format(s))
#                chart1 = chart1 + text
              except KeyError:
                print(s,'is not present in either df_day or sim_dataframe')
#                print('keys sim_dataframe:\t',analysis.sim_dataframe.columns,'\nkeys df_day:\t',sim.df_day.columns)
              d1 = {'difference correlation squared':[],'barrier':[]}
              df1 = pd.DataFrame(d1)
              try:
                df1['difference correlation squared'] = analysis.dif_cor[s]
                df1['barrier']= analysis.dif_cor.columns
              except KeyError:
                print(s,'is not present in analysis.dif_cor')
#                print('keys analysis.dif_cor:\t',analysis.dif_cor.columns)
              chart2 = alt.Chart(df1).mark_line().encode(x='barrier:N', y='difference correlation squared:Q')
              if self.windows:
                if not os.path.exists(sim.state_basename + map_+ 'daily'):
                  os.mkdir(sim.state_basename + map_+ 'daily')
                with open(sim.state_basename + map_+ 'daily/' + 'comparison_sim_real_{}.html', 'w') as f:
                  f.write(two_charts_template.format(
                  vega_version=alt.VEGA_VERSION,
                  vegalite_version=alt.VEGALITE_VERSION,
                  vegaembed_version=alt.VEGAEMBED_VERSION,
                  spec1=chart1.to_json(indent=None),
                  spec2=chart2.to_json(indent=None),
              ))
                html_file = open(sim.state_basename + map_+ 'daily/' + 'comparison_sim_real_{}.html', 'r', encoding='utf-8')

              else:
                if not os.path.exists(sim.state_basename + map_+ 'daily'):
                  os.mkdir(sim.state_basename + map_+ 'daily')                    
                with open(sim.state_basename + map_+ 'daily\\' + 'comparison_sim_real_{}.html', 'w') as f:
                  f.write(two_charts_template.format(
                  vega_version=alt.VEGA_VERSION,
                  vegalite_version=alt.VEGALITE_VERSION,
                  vegaembed_version=alt.VEGAEMBED_VERSION,
                  spec1=chart1.to_json(indent=None),
                  spec2=chart2.to_json(indent=None),
              ))
                html_file = open(sim.state_basename + map_+ 'daily\\' + 'comparison_sim_real_{}.html', 'r', encoding='utf-8')
                  
            elif sim.average_fluxes:
              d = {'sim':[],'real':[],'time':[]}
              df = pd.DataFrame(d)
              try:
                df['sim'] = analysis.sim_dataframe[s][2:]
                df['real_'] = ['real data' for x in range(len(analysis.sim_dataframe[s][2:]))]
                df['sim_'] = ['simulation' for x in range(len(analysis.sim_dataframe[s][2:]))]
                df['real'] = sim.df_avg[s][:-2]
                df['time'] = list(np.arange(22))
                c_sim = alt.Chart(df).mark_line().encode(x='time:N', y='sim:Q',color = 'sim_:N')
                c_real = alt.Chart(df).mark_line().encode(x='time:N', y='real:Q',color = 'real_:N').properties(title = s)#.facet(column = '{}:N'.format(s)).configure_header(titleFontSize=40, labelFontSize=40)
                chart1 = c_sim + c_real
#                text = chart1.mark_text(align='center').encode(text='{}:N'.format(s))
#                chart1 = alt.layer(chart1, text)
              except KeyError:
                print(s,'is not present in either df_avg or sim_dataframe')
#                print('keys sim_dataframe:\t',analysis.sim_dataframe.columns,'\nkeys df_avg:\t',sim.df_avg.columns)
              d1 = {'difference correlation squared':[],'barrier':[]}
              df1 = pd.DataFrame(d1)
              try:
                df1['difference correlation squared'] = analysis.dif_cor[s]
                df1['barrier']= analysis.dif_cor.columns
              except KeyError:
                print(s,'is not present in analysis.dif_cor')
                print('keys analysis.dif_cor:\t',analysis.dif_cor.columns)
              chart2 = alt.Chart(df1).mark_line().encode(x='barrier:N', y='difference correlation squared:Q')
              if self.windows:
                if not os.path.exists(sim.state_basename + map_+ 'averaged'):
                  os.mkdir(sim.state_basename + map_+ 'averaged')
                with open(sim.state_basename + map_+ 'averaged/' +'comparison_sim_real_{}.html'.format(s), 'w') as f:
                  f.write(two_charts_template.format(
                  vega_version=alt.VEGA_VERSION,
                  vegalite_version=alt.VEGALITE_VERSION,
                  vegaembed_version=alt.VEGAEMBED_VERSION,
                  spec1=chart1.to_json(indent=None),
                  spec2=chart2.to_json(indent=None),
              ))
                  
                html_file = open(sim.state_basename + map_+ 'averaged/' +'comparison_sim_real_{}.html'.format(s), 'r', encoding='utf-8')

              else:            
                if not os.path.exists(sim.state_basename + map_+ 'averaged'):
                  os.mkdir(sim.state_basename + map_+ 'averaged')
                with open(sim.state_basename + map_+ 'averaged\\' +'comparison_sim_real_{}.html'.format(s), 'w') as f:
                  f.write(two_charts_template.format(
                  vega_version=alt.VEGA_VERSION,
                  vegalite_version=alt.VEGALITE_VERSION,
                  vegaembed_version=alt.VEGAEMBED_VERSION,
                  spec1=chart1.to_json(indent=None),
                  spec2=chart2.to_json(indent=None),
              ))
                html_file = open(sim.state_basename + map_+ 'averaged\\' +'comparison_sim_real_{}.html'.format(s), 'r', encoding='utf-8')
            charts_code = html_file.read() 
            
            iframe = branca.element.IFrame(html=charts_code, width=1500, height=400)
            print('I am adding at level 0 ',coord_mat[s])
            popup = folium.Popup(iframe, max_width=2000)
            folium.Marker(location = coord_mat[s], popup=popup).add_to(mappa)
#            popup = folium.Popup(s,parse_html = True)
#            folium.Marker(location = coord_mat[s], popup= popup, icon = folium.Icon(color=color)).add_to(mappa)

          except KeyError:
              print('cannot add barrier s',s)
        try:
          if count_barriers !=0 and coord_mat[self.tags[count_barriers]] != coord_mat[self.tags[count_barriers - 1]]:                
            two_charts_template = """
            <!DOCTYPE html>
            <html>
            <head>
              <script src="https://cdn.jsdelivr.net/npm/vega@{vega_version}"></script>
              <script src="https://cdn.jsdelivr.net/npm/vega-lite@{vegalite_version}"></script>
              <script src="https://cdn.jsdelivr.net/npm/vega-embed@{vegaembed_version}"></script>
            </head>
            <body>

            <div id="vis1"></div>
            <div id="vis2"></div>

            <script type="text/javascript">
              vegaEmbed('#vis1', {spec1}).catch(console.error);
              vegaEmbed('#vis2', {spec2}).catch(console.error);
            </script>
            </body>
            </html>
            """
            if sim.pick_day:
              d = {'sim': [],'real':[],'time':[]}
              df = pd.DataFrame(d)
              try:
                df['sim'] = analysis.sim_dataframe[s][2:]
                df['real_'] = ['real data' for x in range(len(analysis.sim_dataframe[s][2:]))]
                df['sim_'] = ['simulation' for x in range(len(analysis.sim_dataframe[s][2:]))]
                df['real'] = sim.df_day[s][:-2]
                df['time'] = list(np.arange(22))
                c_sim = alt.Chart(df).mark_line().encode(x='time:N', y='sim:Q',color = 'sim_:N')
                c_real = alt.Chart(df).mark_line().encode(x='time:N', y='real:Q',color = 'real_:N').properties(title = s)#.properties(width=180,height=180).facet(column = '{}:N'.format(s)).configure_header(titleFontSize=40, labelFontSize=40)
                chart1 = c_sim + c_real
#                text = chart1.mark_text(align='center').encode(text='{}:N'.format(s))
#                alt.layer(chart1, text)

              except KeyError:
                print(s,'is not present in either df_day or sim_dataframe')
#                print('keys sim_dataframe:\t',analysis.sim_dataframe.columns,'\nkeys df_day:\t',sim.df_day.columns)
              d1 = {'difference correlation squared':[],'barrier':[]}
              df1 = pd.DataFrame(d1)
              try:
                df1['difference correlation squared'] = analysis.dif_cor[s]
                df1['barrier']= analysis.dif_cor.columns
              except KeyError:
                print(s,'is not present in analysis.dif_cor')
#                print('keys analysis.dif_cor:\t',analysis.dif_cor.columns)
              chart2 = alt.Chart(df1).mark_line().encode(x='barrier:N', y='difference correlation squared:Q')
              if self.windows:
                if not os.path.exists(sim.state_basename + map_+ 'daily'):
                  os.mkdir(sim.state_basename + map_+ 'daily')
                with open(sim.state_basename + map_+ 'daily/' + 'comparison_sim_real_{}.html', 'w') as f:
                  f.write(two_charts_template.format(
                  vega_version=alt.VEGA_VERSION,
                  vegalite_version=alt.VEGALITE_VERSION,
                  vegaembed_version=alt.VEGAEMBED_VERSION,
                  spec1=chart1.to_json(indent=None),
                  spec2=chart2.to_json(indent=None),
              ))
                html_file = open(sim.state_basename + map_+ 'daily/' + 'comparison_sim_real_{}.html', 'r', encoding='utf-8')

              else:
                if not os.path.exists(sim.state_basename + map_+ 'daily'):
                  os.mkdir(sim.state_basename + map_+ 'daily')
                with open(sim.state_basename + map_+ 'daily\\' + 'comparison_sim_real_{}.html', 'w') as f:
                  f.write(two_charts_template.format(
                  vega_version=alt.VEGA_VERSION,
                  vegalite_version=alt.VEGALITE_VERSION,
                  vegaembed_version=alt.VEGAEMBED_VERSION,
                  spec1=chart1.to_json(indent=None),
                  spec2=chart2.to_json(indent=None),
              ))

                html_file = open(sim.state_basename + map_+ 'daily\\' + 'comparison_sim_real_{}.html', 'r', encoding='utf-8')
                  
            elif sim.average_fluxes:
              d = {'sim':[],'real':[],'time':[]}
              df = pd.DataFrame(d)
              try:
                df['sim'] = analysis.sim_dataframe[s][2:]
                df['real_'] = ['real data' for x in range(len(analysis.sim_dataframe[s][2:]))]
                df['sim_'] = ['simulation' for x in range(len(analysis.sim_dataframe[s][2:]))]
                df['real'] = sim.df_avg[s][:-2]
                df['time'] = list(np.arange(22))
                c_sim = alt.Chart(df).mark_line().encode(x='time:N', y='sim:Q',color = 'sim_:N')
                c_real = alt.Chart(df).mark_line().encode(x='time:N', y='real:Q',color = 'real_:N').properties(title = s)#.properties(width=180,height=180).facet(column = '{}:N'.format(s)).configure_header(titleFontSize=40, labelFontSize=40)
                chart1 = c_sim + c_real
#                text = chart1.mark_text(align='center').encode(text='{}:N'.format(s))
#                alt.layer(chart1, text)
              except KeyError:
                print(s,'is not present in either df_avg or sim_dataframe')
#                print('keys sim_dataframe:\t',analysis.sim_dataframe.columns,'\nkeys df_avg:\t',sim.df_avg.columns)
              d1 = {'difference correlation squared':[],'barrier':[]}
              df1 = pd.DataFrame(d1)
              try:
                df1['difference correlation squared'] = analysis.dif_cor[s]
                df1['barrier']= analysis.dif_cor.columns
              except KeyError:
                print(s,'is not present in analysis.dif_cor')
#                print('keys analysis.dif_cor:\t',analysis.dif_cor.columns)
              chart2 = alt.Chart(df1).mark_line().encode(x='barrier:N', y='difference correlation squared:Q')
              if self.windows:
                if not os.path.exists(sim.state_basename + map_+ 'averaged'):
                  os.mkdir(sim.state_basename + map_+ 'averaged')
                with open(sim.state_basename + map_+ 'averaged/' +'comparison_sim_real_{}.html'.format(s), 'w') as f:
                  f.write(two_charts_template.format(
                  vega_version=alt.VEGA_VERSION,
                  vegalite_version=alt.VEGALITE_VERSION,
                  vegaembed_version=alt.VEGAEMBED_VERSION,
                  spec1=chart1.to_json(indent=None),
                  spec2=chart2.to_json(indent=None),
              ))
                  
                html_file = open(sim.state_basename + map_+ 'averaged/' +'comparison_sim_real_{}.html'.format(s), 'r', encoding='utf-8')

              else:            
                if not os.path.exists(sim.state_basename + map_+ 'averaged'):
                  os.mkdir(sim.state_basename + map_+ 'averaged')
                with open(sim.state_basename + map_+ 'averaged\\' +'comparison_sim_real_{}.html'.format(s), 'w') as f:
                  f.write(two_charts_template.format(
                  vega_version=alt.VEGA_VERSION,
                  vegalite_version=alt.VEGALITE_VERSION,
                  vegaembed_version=alt.VEGAEMBED_VERSION,
                  spec1=chart1.to_json(indent=None),
                  spec2=chart2.to_json(indent=None),))

                html_file = open(sim.state_basename + map_+ 'averaged\\' +'comparison_sim_real_{}.html'.format(s), 'r', encoding='utf-8')
            charts_code = html_file.read() 
            iframe = branca.element.IFrame(html=charts_code, width=1500, height=400)
            print('I am adding',coord_mat[s])
            popup = folium.Popup(iframe, max_width=2000)
            folium.Marker(location = coord_mat[s], popup=popup).add_to(mappa)
#            popup = folium.Popup(s,parse_html = True)
#            folium.Marker(location = coord_mat[s], popup= popup, icon = folium.Icon(color=color)).add_to(mappa)
        except KeyError:
          print('cannot add the barrier')
      if self.windows:              
        mappa.save(sim.state_basename + map_ +'barriers_correlation_distance_plots.html')
      else:
        mappa.save(sim.state_basename + map_ +'barriers_correlation_distance_plots.html')
              
        return True
#################### MAP BEST AND WORST BARRIERS IN and OUT SEPARATELY ################################      
      
    def map_best_worst_euclidean_temporal_distance_in(self,sim,ch,map_,analysis):
      '''Doesn't work very well. The idea was to plot the best and worst correlated barriers'''
      import cmasher as cmr
      print('map_best_worst_euclidean_temporal_distance_in')
      mappa = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)
      coord_mat = {}
      for s in self.tags:
        try:
          coords = list(self.coilsdf[self.coilsdf['Description']==s][['Lat','Lon']].values[0])
          coord_mat[s] = coords
        except IndexError:
          print('the barrier {} is not present in the barriers'.format(s))
#      print(coord_mat)
      for key in list(analysis.dict_df_ward.keys()):
        maxv_, minv_ = np.max(self.dict_df_ward[key]['distance']), np.min(self.dict_df_ward[key]['distance'])
        print('min',minv_,'max',maxv_)
        colormap = colors.LogNorm(vmin=float(minv_), vmax=float(maxv_))
        cmap_ = plt.cm.get_cmap('tab20')
        num = cmap_.N
#        colormap = cm.LinearColormap(colors=[0,1,2], index=[float(minv_),float(maxv_/2),float(maxv_)], vmin=float(minv_), vmax=float(maxv_)) #['blue','yellow','red']
        data = analysis.dict_df_ward[key]
        dataslice = data[:self.topn].copy()
        for cid, row in dataslice.iterrows():
          s,s1 = row['barrier1'], row['barrier2']
          # TODO CONTROL THE LOGIC 
          if not '_OUT' in s and not  '_OUT' in s1:
            name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
            if s in [x.upper() for x in list(ch.dict_sources.keys())]:
              name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
              if not ch.dict_sources[name].is_reset or ch.dict_sources[name].is_added or ch.dict_sources[name].is_changed or ch.dict_sources[name].is_default:
                color = 'red'
            elif s in [x.upper() for x in list(ch.dict_attractions.keys())]:       
              name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
              if not ch.dict_attractions[name].is_reset or ch.dict_attractions[name].is_added or  ch.dict_attractions[name].is_changed or ch.dict_attractions[name].is_default:
                color = 'green'
            else:
                color = 'blue'
            name = s1[0] + s1.split('_')[0][1:].lower() + '_' + s1.split('_')[1] + '_' + s1.split('_')[2]
            if s1 in [x.upper() for x in list(ch.dict_sources.keys())]:
              name = s1[0] + s1.split('_')[0][1:].lower() + '_' + s1.split('_')[1] + '_' + s1.split('_')[2]
              if not ch.dict_sources[name].is_reset or ch.dict_sources[name].is_added or ch.dict_sources[name].is_changed or ch.dict_sources[name].is_default:
                color1 = 'red'
            elif s in [x.upper() for x in list(ch.dict_attractions.keys())]:       
              name = s1[0] + s1.split('_')[0][1:].lower() + '_' + s1.split('_')[1] + '_' + s1.split('_')[2]
              if not ch.dict_attractions[name].is_reset or ch.dict_attractions[name].is_added or  ch.dict_attractions[name].is_changed or ch.dict_attractions[name].is_default:
                color1 = 'green'
            else:
                color1 = 'blue'
            try:
              popup = folium.Popup(s,parse_html = True)
              folium.Marker(location = coord_mat[s], popup= popup, icon = folium.Icon(color=color)).add_to(mappa)
              folium.Marker(location = coord_mat[s1], popup= popup, icon = folium.Icon(color1=color1)).add_to(mappa)
    #          color_ = cmr.take_cmap_colors(cmap_,None , cmap_range=(minv_, maxv_), return_fmt='hex') #(row['distance'] % num)
              folium.PolyLine([coord_mat[s],coord_mat[s1]],weight= row['distance']*100,popup='{}>{}:{}'.format(s,s1,row['distance'])).add_to(mappa) #colormap(row['distance'])
            except KeyError:
              print(s,'\tor\t',s1,'are not there')
        if not os.path.exists(sim.state_basename + map_ + 'IN/'):
              os.mkdir(sim.state_basename + map_ + 'IN/')
        mappa.save(sim.state_basename + map_ +'IN/'+'higher_{}_euclidean_distance_plots_in.html'.format(self.topn))
        mappa.save(sim.state_basename + map_ +'IN/'+'lower_{}_euclidean_distance_plots_in.html'.format(self.topn))

    def map_best_worst_euclidean_temporal_distance_out(self,sim,ch,map_,analysis):
      '''Doesn't work very well. The idea was to plot the best and worst correlated barriers'''
      import cmasher as cmr
      print('map_best_worst_euclidean_temporal_distance_out')
      mappa = folium.Map(location = self.center_coords,tiles='cartodbpositron', control_scale=True, zoom_start=self.zoom)
      coord_mat = {}
      for s in self.tags:
        try:
          coords = list(self.coilsdf[self.coilsdf['Description']==s][['Lat','Lon']].values[0])
          coord_mat[s] = coords
        except IndexError:
          print('the barrier {} is not present in the barriers'.format(s))
#      print(coord_mat)
      for key in list(analysis.dict_df_ward.keys()):
        maxv_, minv_ = np.max(self.dict_df_ward[key]['distance']), np.min(self.dict_df_ward[key]['distance'])
        print('min',minv_,'max',maxv_)
        colormap = colors.LogNorm(vmin=float(minv_), vmax=float(maxv_))
        cmap_ = plt.cm.get_cmap('tab20')
        num = cmap_.N
#        colormap = cm.LinearColormap(colors=[0,1,2], index=[float(minv_),float(maxv_/2),float(maxv_)], vmin=float(minv_), vmax=float(maxv_)) #['blue','yellow','red']
        data = analysis.dict_df_ward[key]
        dataslice = data[:self.topn].copy()
        for cid, row in dataslice.iterrows():
          s,s1 = row['barrier1'], row['barrier2']
          # TODO CONTROL THE LOGIC 
          if '_OUT' in s and '_OUT' in s1:
            name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
            if s in [x.upper() for x in list(ch.dict_sources.keys())]:
              name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
              if not ch.dict_sources[name].is_reset or ch.dict_sources[name].is_added or ch.dict_sources[name].is_changed or ch.dict_sources[name].is_default:
                color = 'red'
            elif s in [x.upper() for x in list(ch.dict_attractions.keys())]:       
              name = s[0] + s.split('_')[0][1:].lower() + '_' + s.split('_')[1] + '_' + s.split('_')[2]
              if not ch.dict_attractions[name].is_reset or ch.dict_attractions[name].is_added or  ch.dict_attractions[name].is_changed or ch.dict_attractions[name].is_default:
                color = 'green'
            else:
                color = 'blue'
            name = s1[0] + s1.split('_')[0][1:].lower() + '_' + s1.split('_')[1] + '_' + s1.split('_')[2]
            if s1 in [x.upper() for x in list(ch.dict_sources.keys())]:
              name = s1[0] + s1.split('_')[0][1:].lower() + '_' + s1.split('_')[1] + '_' + s1.split('_')[2]
              if not ch.dict_sources[name].is_reset or ch.dict_sources[name].is_added or ch.dict_sources[name].is_changed or ch.dict_sources[name].is_default:
                color1 = 'red'
            elif s in [x.upper() for x in list(ch.dict_attractions.keys())]:       
              name = s1[0] + s1.split('_')[0][1:].lower() + '_' + s1.split('_')[1] + '_' + s1.split('_')[2]
              if not ch.dict_attractions[name].is_reset or ch.dict_attractions[name].is_added or  ch.dict_attractions[name].is_changed or ch.dict_attractions[name].is_default:
                color1 = 'green'
            else:
                color1 = 'blue'
            try:
              popup = folium.Popup(s,parse_html = True)
              folium.Marker(location = coord_mat[s], popup= popup, icon = folium.Icon(color=color)).add_to(mappa)
              folium.Marker(location = coord_mat[s1], popup= popup, icon = folium.Icon(color1=color1)).add_to(mappa)
    #          color_ = cmr.take_cmap_colors(cmap_,None , cmap_range=(minv_, maxv_), return_fmt='hex') #(row['distance'] % num)
              folium.PolyLine([coord_mat[s],coord_mat[s1]],weight= row['distance']*100,popup='{}>{}:{}'.format(s,s1,row['distance'])).add_to(mappa) #colormap(row['distance'])
            except KeyError:
              print(s,'\tor\t',s1,'are not there')
        if not os.path.exists(sim.state_basename + map_ + 'OUT/'):
          os.mkdir(sim.state_basename + map_ + 'OUT/')
        mappa.save(sim.state_basename + map_ + 'OUT/'+'higher_{}_euclidean_distance_plots_out.html'.format(self.topn))
        mappa.save(sim.state_basename + map_ + 'OUT/'+'lower_{}_euclidean_distance_plots_out.html'.format(self.topn))

#################### MAP BEST AND WORST BARRIERS IN and OUT SEPARATELY ################################

#################### MAP WARD CLUSTERING BARRIERS IN and OUT SEPARATELY ################################
          
    def map_ward_clustering_in(self,sim,analisys,ch,map_):
      '''Description:
      -------------------------
      Represents in the map, barriers in their cluster. COlors each barrier in the cluster it belongs in all the cases:
      1- sim <- analisys.dict_df_ward['df_dist_sim'],analisys.dict_ncluster['df_dist_sim'],analisys.dict_list_cluster['df_dist_sim']
      2- sim_norm <- analisys.dict_df_ward['df_dist_sim'],analisys.dict_ncluster['df_dist_sim_norm'],analisys.dict_list_cluster['df_dist_sim_norm']
      3- real
      4- real_norm
      This function is defined for IN going people'''
      for k in list(analisys.dict_df_ward.keys()):
        go_with_save = True
        nclusters = analisys.dict_ncluster[k]
        m = folium.Map(location=self.center_coords,tiles='cartodbpositron', control_scale=True,zoom_start=12)
        layerlabel = '<span style="color: {col};">{txt}</span>'
        try:
          flayer = [folium.FeatureGroup(name=layerlabel.format(col=analisys.iconcolors[c], txt=f'cluster {c+1}'), show=True) for c in range(analisys.dict_ncluster[k])]
          for cid, row in analisys.dict_list_cluster[k].iterrows():
            s = row['barrier']
            c = row['cluster']
            if not '_OUT' in s:
              # i = BeautifyIcon(icon=f'{verse}', inner_icon_style=f'color:{iconcolors[c]};font-size:30px;', background_color='transparent', border_color='transparent')
              if s[-1]=='0':
#                i = folium.DivIcon(html=(f'<svg height="150" width="100"> <text x="0" y="35" fill={analysis.iconcolors[c]}>0</text> </svg>'))
                i = folium.DivIcon(html=(f'<svg height="150" width="100"> <text x="10" y="50" fill={analysis.iconcolors[c]}>0</text> </svg>'))
              else:
               i = folium.DivIcon(html=(f'<svg height="150" width="100"> <text x="10" y="50" fill={analisys.iconcolors[c]}>1</text> </svg>'))
# i = folium.DivIcon(html=(f'<svg height="150" width="100"> <text x="6" y="35" fill={analisys.iconcolors[c]}>1</text> </svg>'))
              try:
                shp = folium.Marker(location=self.coord_mat[s], popup=s, icon=folium.Icon(color=analisys.iconcolors[c]))
                flayer[c].add_child(shp)
              except:
                print(s,'is not in coord_mat')
            else:
              pass
        except (IndexError,TypeError):
          print('colors are too few, or type dict_cluster[k] = None, number of colors:\t ',len(analisys.iconcolors),'number of cluster',nclusters,'\tcase\t',k)
          go_with_save = False
        if go_with_save:
          for l in flayer: 
            m.add_child(l)
          folium.map.LayerControl(collapsed=False).add_to(m)
          if not os.path.exists(sim.state_basename + map_ + 'IN/'):
            os.mkdir(sim.state_basename + map_ + 'IN/')
          m.save(sim.state_basename + map_ + 'IN/'+'map_ward_clustering_{}_in.html'.format(k))
        else:
            print('I havent saved {} in IN'.format(k))
            pass
          
          

    def map_ward_clustering_out(self,sim,analisys,ch,map_):
      '''Description:
      -------------------------
      Represents in the map, barriers in their cluster. COlors each barrier in the cluster it belongs in all the cases:
      1- sim <- analisys.dict_df_ward['df_dist_sim'],analisys.dict_ncluster['df_dist_sim'],analisys.dict_list_cluster['df_dist_sim']
      2- sim_norm <- analisys.dict_df_ward['df_dist_sim'],analisys.dict_ncluster['df_dist_sim_norm'],analisys.dict_list_cluster['df_dist_sim_norm']
      3- real
      4- real_norm
      This function is defined for OUT going people'''
      for k in list(analisys.dict_df_ward.keys()):
        go_with_save = True
        m = folium.Map(location=self.center_coords,tiles='cartodbpositron', control_scale=True,zoom_start=12)
        nclusters = analisys.dict_ncluster[k]
        layerlabel = '<span style="color: {col};">{txt}</span>'
        try:
          flayer = [folium.FeatureGroup(name=layerlabel.format(col=analisys.iconcolors[c], txt=f'cluster {c+1}'), show=True) for c in range(analisys.dict_ncluster[k])]

          for cid, row in analisys.dict_list_cluster[k].iterrows():
            s = row['barrier']
            c = row['cluster']
            if '_OUT' in s:        
              # i = BeautifyIcon(icon=f'{verse}', inner_icon_style=f'color:{iconcolors[c]};font-size:30px;', background_color='transparent', border_color='transparent')
              if s[-1]=='0':
                i = folium.DivIcon(html=(f'<svg height="50" width="50"> <text x="0" y="35" fill={analysis.iconcolors[c]}>0</text> </svg>'))
              else:
                i = folium.DivIcon(html=(f'<svg height="50" width="50"> <text x="6" y="35" fill={analisys.iconcolors[c]}>1</text> </svg>'))
              try:
                shp = folium.Marker(location=self.coord_mat[s], popup=s, icon=folium.Icon(color=analisys.iconcolors[c]))
                flayer[c].add_child(shp)
              except:
                print(s,'is not in coord_mat')
            else:
              pass
        except (IndexError,TypeError):
          print('colors are too few, or type dict_cluster[k] = None, number of colors:\t ',len(analisys.iconcolors),'number of cluster',nclusters,'\tcase\t',k)
          go_with_save = False
        if go_with_save:  
          for l in flayer: 
            m.add_child(l)
          folium.map.LayerControl(collapsed=False).add_to(m)
          if not os.path.exists(sim.state_basename + map_ + 'OUT/'):
            os.mkdir(sim.state_basename + map_ + 'OUT/')
          m.save(sim.state_basename + map_ +'OUT/'+'map_ward_clustering_{}_out.html'.format(k))
        else:
          print('I havent saved {} in IN'.format(k))

        
        
#################### MAP WARD CLUSTERING BARRIERS IN and OUT SEPARATELY ################################        
              
        
        
        
        
        
      def map_best_worst_correlation(self,sim,ch,map_,analysis):
        ['SCALZI_2_IN']
        ['SCALZI_3_IN']
        ['SCALZI_2_IN','SCALZI_3_IN']
        ['SCALZI_2_IN','COSTITUZIONE_IN']
        ['SCALZI_2_IN','PAPADOPOLI_IN']
        ['SCALZI_3_IN','COSTITUZIONE_IN']
        ['SCALZI_3_IN','PAPADOPOLI_IN']
        ['PAPADOPOLI_IN','COSTITUZIONE_IN']
        ['SCALZI_2_IN','COSTITUZIONE_IN']
        ['SCALZI_2_IN','SCALZI_3_IN','PAPADOPOLI_IN','COSTITUZIONE_IN']

        maxv, minv = np.max(data[ordtype].values), np.min(data[ordtype].values)
        colormap = cm.LinearColormap(colors=['blue','yellow','red'], index=[minv,maxv/2,maxv], vmin=minv, vmax=maxv)

        # top N
        dataslice = data[:topn].copy()
        mappa = folium.Map(location=center_coords, tiles='cartodbpositron', control_scale=True, zoom_start=9)
        for cid, row in dataslice.iterrows():
          inp,out = row['barrier1'], row['barrier2']
          folium.Marker(location=coord_mat[inp], popup=inp).add_to(mappa)
          folium.Marker(location=coord_mat[out], popup=out).add_to(mappa)
          folium.PolyLine([coord_mat[inp],coord_mat[out]],color=colormap(row[ordtype]),weight=2,popup='{}>{}:{}'.format(inp,out,row[ordtype])).add_to(mappa)

        mappa.add_child(colormap)
        mappa.save(sim.state_basename + map_ +'higher_{}_correlation_plots.html').format(topn)
        mappa.save(sim.state_basename + map_ +'lower_{}_correlation_plots.html').format(topn)


      def map_best_worst_ward_distance(self,sim,ch,map_,analysis):


        mappa.save(sim.state_basename + map_ +'higher_{}_ward_distance_plots.html').format(topn)
        mappa.save(sim.state_basename + map_ +'lower_{}_ward_distance_plots.html').format(topn)

            

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

