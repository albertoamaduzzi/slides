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
#import dtw
import os
import argparse

#Enter parameters start_,end_ date of partitioning by ward clusterings
parser=argparse.ArgumentParser(description='Insert starting and ending date of the simulation period wanted in form %Y-%m-%d %H:%M:%S default 2021-07-15, or avg_opt= True to obtain ward for averaged data')
parser.add_argument('--start_date',help='insert start date in format %Y-%m-%d %H:%M:%S, default:2021-07-15 00:00:00 ',type=str,default='2021-07-15 00:00:00')
parser.add_argument('--stop_date',help='insert start date in format %Y-%m-%d %H:%M:%S, default:2021-07-15 23:59:59',type=str,default='2021-07-15 23:59:59')
parser.add_argument('--avg_opt',help='insert average',type=bool,default=False)
parser.add_argument('--dffile',help='insert the file to be analyzed',type=str,default=r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\data\venezia_no_locals_barriers_210715_000000.csv')

args= parser.parse_args()




#%%
def norm(a):
  tot = a.sum()
  if tot!=0: return a/tot
  else:      return a

def dist(a,b,t='euclidean'):
  if t == 'euclidean':
    return np.sqrt(np.sum((a-b)**2)) # or   ssd.euclidean(a,b)
#  elif t == 'correlation':
#    return ssd.correlation(a,b)
#    return dtw.dtw(a,b,distance_only=True).distance
    # with radius 8 and 15min data interval there is a 2 hour range for dtw

def inv_barriers(s):
  lc = s[-1]
  if lc == '0': return s[:-1]+'1'
  else:         return s[:-1]+'0'

#%% INPUT
working_dir =r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\data'
dffile = args.dffile #marzo 2022 Albi#
#dffile = 'coils_table_200201-000000_200331-234500.csv' # febbraio-marzo 2020
# dffile = 'coils_table_210331-220000_210430-214500.csv' # aprile 2021
# dffile = 'coils_table_210430-220000_210531-214500.csv' # maggio 2021


#%% OUTPUT
if args.avg_opt==True:
  if not os.path.exists(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\output\averaged'):
    os.mkdir(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\output\averaged')
  saving_dir =r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\output\averaged'
else:
  if not os.path.exists(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\output\daily'):
    os.mkdir(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\output\daily')
  saving_dir =r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\output\daily'

dist_type = 'euclidean' # norma L2
# dist_type = 'correlation' # distanza di correlazione
# dist_type = 'dtw' # dynamic time warping distance

hca_method = 'ward'
# hca_method = 'complete'

nan_t = 5 # %. droppa le barriers con troppi nan
sm_window = 9 # smoothing window, deve essere dispari
sm_order = 3 # smoothing polynomial order
use_smooth = True # usare i dati smooth o i dati raw
normalize = True # normalizzare i dati per giornata

# plt.ioff()

df = pd.read_csv(dffile, sep=';')
nan_thresh = len(df)*nan_t/100
tagn_orig = len(df.columns) -1
"""
# usa solo le barriers utilizzate nel modello
dffbarriers = 'coils-dir-mapping_template.csv'
dfbarriers = pd.read_csv(dffbarriers,sep=';')
dfbarriers['tag'] = dfbarriers['coil_id'].astype(str)+'_'+dfbarriers['dir_reg'].astype(str)
tags_sp = list(dfbarriers['tag'])
df = df[['datetime']+tags_sp]
"""

# trova le barriers con troppi nan
drop_list = []
for c in df.columns:
    if df[c].isna().sum()>nan_thresh:
        drop_list.append(c)
# trova anche le barriers nel verso opposto
for s in drop_list:
  invs = inv_barriers(s)
  if (invs not in drop_list) and (invs in df.columns) : drop_list.append(invs)
df = df.drop(drop_list, axis=1)

df = df.fillna(0)
df['datetime'] = pd.to_datetime(df['datetime'])

tags = list(df.columns.drop(['datetime','timestamp']))
tagn = len(tags)
print(f'Using {tagn} / {tagn_orig} barriers')

#smoothing
if use_smooth:
  dfsm = df.apply(lambda x: savgol_filter(x,sm_window,sm_order) if x.name in tags else x)
  dfc = dfsm
  """
  # smooth plot example
  plt.title('Smoothing example for barriers 636_2')
  plt.plot(df['datetime'],df['636_2'],'-r')
  plt.plot(dfc['datetime'],dfc['636_2'],'-b')
  plt.legend(['raw','smooth'])
  """
else:
  dfc = df

group = dfc.groupby(pd.Grouper(key='datetime', freq='D'))

#%% DISTANCE MATRIX

dist_mat = []

for day, dfg in group:
  t_mat = []
  print('Elaborating '+day.strftime('%d-%m-%Y'))
  for j in range(0,tagn-1):
    for k in range(j+1,tagn):
      if normalize:
        distance = dist(norm(dfg[tags[j]].values),norm(dfg[tags[k]].values),dist_type)
      else:
        distance = dist(dfg[tags[j]].values,dfg[tags[k]].values,dist_type)
      t_mat.append([distance,j,k])
      '''      
      print(t_mat,np.shape(t_mat))
[[0.022324633566389102, 0, 1]] (1, 3)
[[0.022324633566389102, 0, 1], [0.026243855660838318, 0, 2]] (2, 3)
[[0.022324633566389102, 0, 1], [0.026243855660838318, 0, 2], [0.04640471908383456, 0, 3]] (3, 3)
[[0.022324633566389102, 0, 1], [0.026243855660838318, 0, 2], [0.04640471908383456, 0, 3], [0.012332751402646093, 0, 4]] (4, 3)
[[0.022324633566389102, 0, 1], [0.026243855660838318, 0, 2], [0.04640471908383456, 0, 3], [0.012332751402646093, 0, 4], [0.02542105324950935, 0, 5]] (5, 3)        '''
  dist_mat.append(t_mat)
dist_mat = np.array(dist_mat) # pairwise long form distance matrix

dfdevs = pd.DataFrame()
dfdevs['tag1']= dist_mat[0,:,1].astype(int)
dfdevs['tag2']= dist_mat[0,:,2].astype(int)#le due colonne insieme mi parlano della distanza devo fare un groupby sulle due colonne per 
dfdevs['barrier1'] = [tags[int(a)] for a in dfdevs['tag1'].values]
dfdevs['barrier2'] = [tags[int(a)] for a in dfdevs['tag2'].values]
dfdevs['std'] = np.std(dist_mat,axis=0)[:,0]
dfdevs['sum'] = np.sum(dist_mat,axis=0)[:,0]

# add all columns before sorting
dfstds = dfdevs.sort_values(by='std')
dfsums = dfdevs.sort_values(by='sum')

dfdevs.to_csv(os.path.join(saving_dir,'df_barriers_dist.csv'))
#%% PLOTS
pos = 0 # Nth best pair
ordtype = 'sum' # sum or std?

titolo = f'Barriers countings, ordered by curve distance {ordtype}, '

if   ordtype == 'sum': data = dfsums
elif ordtype == 'std': data = dfstds
if normalize: titolo+= 'normalized, '
if use_smooth: titolo+= 'smooth, '

barrier1 = data['barrier1'].iloc[pos]
barrier2 = data['barrier2'].iloc[pos]
print('barrier1',barrier1,'barrier2',barrier2)
titolo+= f'{pos+1}° best pair: {barrier1} / {barrier2}'

fig = plt.figure(figsize=(12, 7))
if normalize:
  plt.plot(dfc.datetime.values, norm(dfc[barrier1]), 'r', alpha=0.8)
  plt.plot(dfc.datetime.values, norm(dfc[barrier2]), 'b', alpha=0.8)
else:
  plt.plot(dfc.datetime.values, dfc[barrier1], 'r', alpha=0.8)
  plt.plot(dfc.datetime.values, dfc[barrier2], 'b', alpha=0.8)
plt.title(titolo)
plt.xlabel('Date')
plt.ylabel('Countings')
plt.legend([barrier1,barrier2])
#plt.show()
# Mi da 2 curve che si sovrappongono temporalmente MANDOLA_1,4 mostra sulle x l'orario e sulle y countings
#%% sum vs std plots

# plt.title('barriers pairs daily curve sum')
# plt.plot(dfsums['sum'].values,'-b', linewidth=1)
plt.title('Normalized daily curve distances sum vs std')
plt.plot(norm(dfsums['std'].values),',r')
plt.plot(norm(dfsums['sum'].values),'-b')
plt.legend(['std','sum'])
plt.xlabel('barriers pair ordered by sum')
#plt.show()
#I cannot understand this. On the x I have years from 1990 to 2020
#%% geoplot
# top N link and worst N link
topn = 30

# setup
coilsdf = pd.read_csv(os.path.join(working_dir,'barriers_config.csv'), sep=';')
center_coords = coilsdf[['Lat', 'Lon']].mean().values
coord_mat = {}
c=0
for s in tags:
  coilid = s#.split('_')[0]
  if len(coilsdf[coilsdf['Description']==coilid][['Lat','Lon']].values)==0:
    name=s.split('_')[0]
    for s1 in tags:
#      print(s1,s1.find(name))
      if (s1.find(name)!= -1) and s1!=s:
        if len(coilsdf[coilsdf['Description']==s1][['Lat','Lon']].values)!=0:
          coords =  list(coilsdf[coilsdf['Description']==s1][['Lat','Lon']].values[0])
#          print('s1',s1)
          break
  else:
    coords = list(coilsdf[coilsdf['Description']==coilid][['Lat','Lon']].values[0])
#  print(coords)
#  print(coilsdf[coilsdf['Description']==coilid][['Lat','Lon']].values,'\t',s,'\t',c)
  coord_mat[s] = coords
#  c=c+1

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
mappa.save(os.path.join(saving_dir,f'barriers_top_{topn}.html'))
# worst N
mappa = folium.Map(location=center_coords, tiles='cartodbpositron', control_scale=True, zoom_start=9)
for cid, row in data[-topn:].iterrows():
  inp,out = row['barrier1'], row['barrier2']
  folium.Marker(location=coord_mat[inp], popup=inp).add_to(mappa)
  folium.Marker(location=coord_mat[out], popup=out).add_to(mappa)
  folium.PolyLine([coord_mat[inp],coord_mat[out]],color=colormap(row[ordtype]),weight=2,popup='{}>{}:{}'.format(inp,out,row[ordtype])).add_to(mappa)

mappa.add_child(colormap)
mappa.save(os.path.join(saving_dir,f'barriers_worst_{topn}.html'))

#%% check delle barriers nella top che hanno link in tutte le variazioni possibli
unilist = []
for cid, row in dataslice.iterrows():
  inp,out = row['barrier1'], row['barrier2']
  unitag =  inp[:-2]+'-'+out[:-2]
  if unitag not in unilist:
    invi, invo = inv_barriers(inp), inv_barriers(out)
    con1 = (dataslice[['barrier1','barrier2']]==np.array([inp,invo])).all(1).any()
    con2 = (dataslice[['barrier1','barrier2']]==np.array([invi,out])).all(1).any()
    con3 = (dataslice[['barrier1','barrier2']]==np.array([invi,invo])).all(1).any()
    if con1 and con2 and con3:
      unilist.append(unitag)

print('List of all verse correlated barrierss: ', unilist)

mappa = folium.Map(location=center_coords,tiles='cartodbpositron',control_scale=True,zoom_start=9)
for pair in unilist:
  inp,out = pair.split('-')
  folium.Marker(location=coord_mat[inp+'_0'], popup=folium.Popup(inp,show=True)).add_to(mappa)
  folium.Marker(location=coord_mat[out+'_0'], popup=folium.Popup(out,show=True)).add_to(mappa)
  folium.PolyLine([coord_mat[inp+'_0'],coord_mat[out+'_0']],color=colormap(row[ordtype]),weight=2).add_to(mappa)

mappa.add_child(colormap)
mappa.save(os.path.join(saving_dir,'barriers_top_allverse.html'))

#%% migliori/peggiori barriers come correlazione in generale
titolo = 'barriers total distance from other barrierss, data '
if use_smooth: titolo+= 'smooth '
if normalize: titolo+= 'normalized '

square_mat = ssd.squareform(dfdevs[ordtype].values) # square symmetric form distance matrix
best_barriers =[a.sum() for a in square_mat] # somma di tutte le differenze rispetto alle altre barriers
df_best = pd.DataFrame({'barrier':tags,'tot':best_barriers})
df_best = df_best.sort_values(by='tot')
df_best.to_csv(os.path.join(saving_dir,'df_barriers_tot.csv'))

plt.plot(df_best['barrier'],df_best['tot'],'.b')
plt.tight_layout()
plt.title(titolo)
plt.xticks(fontsize=7,rotation=90)
plt.grid()
#plt.show()

mappa = folium.Map(location=center_coords,tiles='cartodbpositron', control_scale=True,zoom_start=9)
for s in df_best.iloc[-10:]['barrier']:
  folium.Marker(location=coord_mat[s], popup=folium.Popup(s,show=True)).add_to(mappa)
mappa.save('barriers_worst_singles_10.html')

###############################################################################
#%% clustering hca
iconcolors = ['purple', 'orange', 'red', 'green', 'blue', 'pink', 'beige', 'darkred', 'darkpurple', 'lightblue', 'lightgreen', 'cadetblue', 'lightgray', 'gray', 'darkgreen', 'white', 'darkblue', 'lightred', 'black']
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111)

linkg = sch.linkage(dfdevs[ordtype].values,method=hca_method)
sch.dendrogram(linkg, labels=tags)
plt.title(f'HCA dendrogram method: {hca_method}, distance: {dist_type}')
plt.xticks(fontsize=7)
plt.savefig(os.path.join(saving_dir,f'HCA dendrogram method_{hca_method}_distance_{dist_type}'))
#plt.show()

#%%
cut_distance = 6 # distanza nel dendrogramma a cui tagliare per decidere il numero di cluster
clusterlist = sch.fcluster(linkg, cut_distance, criterion='distance')-1
df_cluster = pd.DataFrame({'barrier':tags,'cluster':clusterlist})
nclusters = len(np.unique(clusterlist))

assert nclusters <= len(iconcolors) # troppi pochi colori per tutti i cluster altrimenti

m = folium.Map(location=center_coords,tiles='cartodbpositron', control_scale=True,zoom_start=9)
folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', name='satellite', attr='1').add_to(m)
folium.TileLayer('openstreetmap').add_to(m)
folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('Stamen Toner').add_to(m)
folium.TileLayer('Stamen Water Color').add_to(m)
folium.TileLayer('cartodbpositron').add_to(m)
folium.TileLayer('cartodbdark_matter').add_to(m)
folium.TileLayer(tiles='http://a.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', name='osm2', attr='1').add_to(m)
folium.TileLayer(tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', name='topomap', attr='1').add_to(m)
folium.TileLayer(tiles='https://{s}.tile.thunderforest.com/transport-dark/{z}/{x}/{y}.png', name='transport dark', attr='1').add_to(m)
folium.TileLayer(tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain-background/{z}/{x}/{y}{r}.png', name='terrain background', attr='1').add_to(m)

layerlabel = '<span style="color: {col};">{txt}</span>'
flayer = [folium.FeatureGroup(name=layerlabel.format(col=iconcolors[c], txt=f'cluster {c+1}'), show=True) for c in range(nclusters)]

for cid, row in df_cluster.iterrows():
  s = row['barrier']
  c = row['cluster']
  # i = BeautifyIcon(icon=f'{verse}', inner_icon_style=f'color:{iconcolors[c]};font-size:30px;', background_color='transparent', border_color='transparent')
  if s[-1]=='0':
    i = folium.DivIcon(html=(f'<svg height="50" width="50"> <text x="0" y="35" fill={iconcolors[c]}>0</text> </svg>'))
  else:
    i = folium.DivIcon(html=(f'<svg height="50" width="50"> <text x="6" y="35" fill={iconcolors[c]}>1</text> </svg>'))
  shp = folium.Marker(location=coord_mat[s], popup=s, icon=i)
  flayer[c].add_child(shp)

for l in flayer: m.add_child(l)
folium.map.LayerControl(collapsed=False).add_to(m)
m.save(os.path.join(saving_dir,'barriers_all.html'))


#%% grafico di singola barrier
#barrierp = '288_0'
#plt.plot(dfc.datetime.values, df[barrierp], 'b')
#plt.title(f'barriers {barrierp} counting')
#plt.xlabel('Date')
#plt.ylabel('Countings')
#plt.grid()

#%% confronto con dati nuovi
'''
df_old = pd.read_csv('df_barriers_tot_old.csv')
df_new = pd.read_csv('df_barriers_tot_new.csv')
idxoldl = []
idxnewl = []
for s in df_old.barrier.values:
  idxoldl.append(df_old.index[df_old.barrier==s][0]/len(df_old))
  if s in df_new.barrier.values:   idxnewl.append(df_new.index[df_new.barrier==s][0]/len(df_new))
  else: idxnewl.append(-1)

plt.plot(df_old.barrier.values, idxoldl, '.b')
plt.plot(df_old.barrier.values, idxnewl, '.r')
plt.xticks(fontsize=7,rotation=90)
plt.grid()
plt.title('Old vs new data normalized index in total distance order')
'''
#%% locality map
#barrierp = '174_1' # barrier centrale
#kil_radius = 15 # semilato del box in km

#p_latlon = coord_mat[barrierp]
#p_cen = geopy.Point(p_latlon)
#p_dist = geopy.distance.distance(kilometers=kil_radius)
#p_NE = p_dist.destination(point=p_cen, bearing=45) # 0 gradi = nord, senso orario
#p_SW = p_dist.destination(point=p_cen, bearing=225)
#box_latmin, box_latmax, box_lonmin, box_lonmax = p_SW.latitude, p_NE.latitude, p_SW.longitude, p_NE.longitude

box_barriers = []
# check quali barriers sono nel box
for barrier in coord_mat:
  if (box_latmin < coord_mat[barrier][0] < box_latmax) and (box_lonmin < coord_mat[barrier][1] < box_lonmax):
    box_barriers.append(barrier)
if barrierp in box_barriers: box_barriers.remove(barrierp)

mappa = folium.Map(location=p_latlon,tiles='cartodbpositron',control_scale=True,zoom_start=9)
folium.Marker(location=p_latlon, popup=folium.Popup(barrierp,show=True)).add_to(mappa)
for barrier in box_barriers:
  folium.Marker(location=coord_mat[barrier], popup=folium.Popup(barrier,show=True)).add_to(mappa)
  link = data[((data['barrier2']==barrierp) & (data['barrier1']==barrier)) | ((data['barrier1']==barrierp) & (data['barrier2']==barrier))][ordtype].values
  folium.PolyLine(
    [p_latlon,coord_mat[barrier]],
    color=colormap(link),
    weight=2
  ).add_to(mappa)

mappa.add_child(colormap)
mappa.save(os.path.join(saving_dir,f'barriers_locality_{barrierp}.html'))