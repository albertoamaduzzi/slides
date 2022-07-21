#! /usr/bin/env python3
import sys
import os
import json
import argparse
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')   

###########example bash command##############
# cd ../slides/scripts
# python3 test.py -a True -l True
# In this way in the list of change of parameters, that define the number of threads I will have no locals and change of attractions
#


#Enter parameters start_,end_ date of simulation
parser=argparse.ArgumentParser(description='Insert starting and ending date of the simulation in form %Y-%m-%d %H:%M:%S default 2021-07-15')
# Select day and city, select to average over the week
parser.add_argument('--start_date',help='insert start date in format %Y-%m-%d %H:%M:%S, default:2021-07-14 23:00:00 ',type=str,default='2021-07-14 23:00:00')
parser.add_argument('--stop_date',help='insert start date in format %Y-%m-%d %H:%M:%S, default:2021-07-15 23:59:59',type=str,default='2021-07-15 23:59:59')
parser.add_argument('--citytag',help='insert citytag',type=str,default='venezia')
parser.add_argument('-af','--averaging_fluxes',help='True if I want to average sources over the week',type=bool,default=False)
# Dealing changes in the attractions
parser.add_argument('-a','--attraction_activate',help='insert bool to activate changes in the attractions',type=bool,default=False)
# Dealing locals
parser.add_argument('-l', '--locals',help='insert locals true in the case of no_locals',type=bool,default=True)
parser.add_argument('-ld','--local_distribution',help='insert distribution',type=str,default='none')
# Dealing with sources: add new ones, delete older ones
parser.add_argument('-ns','--new_source_list',help='List of names of the sources in conf_barriers.csv to add separated by -',type=str,default='Scalzi_2-Scalzi_3')
parser.add_argument('-nsb','--new_source_bool',help='True if a source has numbered names, i.e. (Scalzi_2_IN-Scalzi_3_IN) default = True',type=bool,default= True)

parser.add_argument('-rs','--reset_source_list',help='Name of the sources in considered for the simulation',type=str,default='Costituzione_IN-Papadopoli_IN-Schiavoni_IN')
parser.add_argument('-rsb','--reset_source_bool',help='True if reset a source, default = True',type=bool,default= True)

parser.add_argument('-cs','--change_source_list',help='Name of the sources in considered for the simulation',type=str,default='Papadopoli_IN')
parser.add_argument('-csb','--change_source_bool',help='True if modify a source, default = True',type=bool, default= True)

parser.add_argument('-na','--name_attractions',help='Name of the attractions added in the simulation',type=str,default='Farsetti_1')
parser.add_argument('-aab','--add_attractions_bool',help='True if add the attractions, default = True',type=bool, default= True)

#parser.add_argument('-lsa','--list_sources_activated',help='Name of the sources in considered for the simulation',type=str,default='Scalzi_2-Scalzi_3')

args= parser.parse_args()

#%%
def pick_day_input(df,start_date):
  '''df: is the file excel whose columns are: timestamp, varco, in,out.
  returns a dataframe whose columns are the sources and rows date-timed fluxes of the day that starts at start_date'''
  time_format='%Y-%m-%d %H:%M:%S'
#  ar_varc=np.unique(np.array(df['varco'],dtype=str))
  group=df.groupby('varco')
  df_new=pd.DataFrame()
  for g in group.groups:
    df_temp=group.get_group(g)
    try:
      df_new['datetime']=np.array(df_temp['timestamp'].copy(),dtype='datetime64[s]')
      df_new[df_temp.iloc[0]['varco']+'_IN']=np.array(df_temp['direzione'].copy(),dtype=int)
      df_new[df_temp.iloc[0]['varco']+'_OUT']=np.array(df_temp['Unnamed: 3'].copy(),dtype=int)  
    except:
      pass
  del df_temp
  start_date = datetime.strptime(start_date,time_format)
  mask_day = [True if h.day==start_date.day else False for h in df_new.datetime]
  df_new = df_new.loc[mask_day]
  print('df new',df_new)
  return df_new

#%%  
def average_fluxes(df):
#  ar_varc = np.unique(np.array(df['varco'],dtype=str))
  group=df.groupby('varco')
  df_new=pd.DataFrame()
  for g in group.groups:
    df_temp=group.get_group(g)
    try:
        df_new['datetime']=np.array(df_temp['timestamp'].copy(),dtype='datetime64[s]')
        df_new[df_temp.iloc[0]['varco']+'_IN']=np.array(df_temp['direzione'].copy(),dtype=int)
        df_new[df_temp.iloc[0]['varco']+'_OUT']=np.array(df_temp['Unnamed: 3'].copy(),dtype=int)  
    except:
      pass
  del df_temp
  df=df_new
  h_mask = np.arange(25)
  tags = list(df.columns.drop('datetime'))
  tagn = len(tags)
  df_avg = pd.DataFrame(data=np.zeros([25,tagn+1]),columns=df.columns)
#  df_avg.set_index(keys=h_mask)
#  print('mask and index',df_avg.index)
  week_mask = [True if h.weekday()<4 else False for h in df.datetime]
  dfc_temp = df.loc[week_mask]
  for e in h_mask:
    mask = [True if h.hour==e else False for h in dfc_temp.datetime]                                          
    df_avg.iloc[e][tags] = dfc_temp[tags].loc[mask].mean()
    df_avg.iloc[e]['datetime'] = e
  return df_avg

def set_simcfg_fluxes_from_list_sources(df_avg,simcfgorig,list_name_sources):
  '''Mi sto riferendo nel caso d'uso a questa lista in input ['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN','SCALZI_1_IN']'''
  if args.averaging_fluxes:
    df_avg =df_avg[:-1]
  else:
    pass
  list_fluxes_conf_file = []
  for name_source in list_name_sources:
    list_flux_single_source = []
    print('extract sources fluxes name source',name_source)
    for flux in df_avg[name_source]:
      for i in range(4):
        list_flux_single_source.append(flux/4)
    if len(list_flux_single_source)!= 96:
      print('la lunghezza della lista per la sorgente {0} è {1}'.format(name_source,len(list_flux_single_source)))
      exit( )
    list_fluxes_conf_file.append(list_flux_single_source)
    if name_source in ['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN','SCALZI_1_IN']:
      name_source = name_source.split('_')[0]+'_'+ name_source.split('_')[2]
    else:
      pass
    new_source_name = ''
    for position_letter in range(len(name_source)):
      if position_letter == 0 or position_letter ==len(name_source)-1 or position_letter ==len(name_source)-2:
        new_source_name = new_source_name + name_source[position_letter]
      else:
        new_source_name = new_source_name + name_source[position_letter].lower() 
    simcfgorig['sources'][new_source_name]['creation_rate'] = list_flux_single_source
    print('extract sources fluxes simcfgorig',simcfgorig['sources'][new_source_name]['creation_rate'])
    return simcfgorig
  
def add_source(df_avg,simcfgorig,name_source):
  ''' adds directly the sources from list_flux_single_source that contains the fluxes of df_avg, the one with columns all the sources.'''
  data_barriers = pd.read_csv(dir_data,';')
  list_flux_single_source = []
  for flux in range(len(df_avg[name_source.upper() + '_IN'])):
    for i in range(4):
      list_flux_single_source.append(list(df_avg[name_source.upper() + '_IN'])[flux]/4)
  if len(list_flux_single_source)!= 96:
    print('la lunghezza della lista per la sorgente {0} è {1}'.format(name_source,len(list_flux_single_source)))
    exit( )  
  simcfgorig['sources'][name_source + '_IN'] = {
        'creation_dt' : simcfgorig['sources']['Costituzione_IN']['creation_dt'],
        'creation_rate' : list_flux_single_source,
        'source_location' : {
          'lat' : data_barriers.loc[data_barriers['Description']==name_source.upper()+'_IN'].iloc[0]['Lat'],
          'lon' : data_barriers.loc[data_barriers['Description']==name_source.upper()+'_IN'].iloc[0]['Lon']
        },
        'pawns_from_weight': {
          'tourist' : {
            'beta_bp_miss' : simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['beta_bp_miss'],
            'speed_mps'    : simcfgorig['sources']['Costituzione_IN']['pawns_from_weight']['tourist']['speed_mps']
          }
        }
      }
  return True
def reset_source(simcfgorig,reset_source):
      list_zero=np.zeros(96).tolist()
      simcfgorig['sources'][reset_source]['creation_rate'] = list_zero
      return True

def sources_simulation_with_people(simcfgorig):      
      intersect =set(args.reset_source_list.split('-')).symmetric_difference(set(simcfgorig['sources'].keys()))
      intersect.discard('LOCALS')
      list_sources = '-'.join(intersect)
      print('new list of sources',list_sources)
      return list_sources

    
def create_list_delta_u_attraction(args):
  attraction_perturbation_activate=args.attraction_activate
  locals_bool = args.locals
  if attraction_perturbation_activate: 
    delta_u = [0.3, 0.25,-0.05,-0.1,-0.15, -0.2,-0.25,-0.3]
    attractions = list(simcfgorig['attractions'].keys())# I am returned with the list of attractions
  else:
    delta_u = [0]
    attractions = list(simcfgorig['attractions'].keys())# I am returned with the list of attractions

  # modifico i parametri di interesse (attractivity) 
  list_delta_u_attraction=[]
  for attraction in attractions:
      for w in delta_u:
            list_delta_u_attraction.append([attraction,w])
  return list_delta_u_attraction

def setting_state_base_name(att_act,locals_bool,new_source_bool,list_delta_u_attraction,simcfgorig,layer_directory_to_add):
# Initialising the change of the attractions
  if att_act:
    simcfgorig['attractions'][attraction]['weight'] = list(np.array(simcfgorig['attractions'][attraction]['weight']) + np.ones(len(np.array(simcfgorig['attractions'][attraction]['weight'])))*w)
#HANDLING CASES
  if locals_bool and att_act:
#
    if  list_delta_u_attraction[1]>0:
      saving_file = args.citytag + list_delta_u_attraction[0] + str(list_delta_u_attraction[1]).split('.')[1]+'_no_locals'         
    else:
          saving_file = args.citytag + list_delta_u_attraction[0] +'_'+str(list_delta_u_attraction[1]).split('.')[1]+'_no_locals'
    if new_source_bool:
      if not os.path.exists(os.path.join(state_basename,layer_directory_to_add)):
        os.mkdir(os.path.join(state_basename,layer_directory_to_add))
    simcfgorig['state_basename'] = os.path.join(state_basename,layer_directory_to_add,saving_file)
#
  elif locals_bool and not att_act:
    print('I am in the case no locals and no variation of attraction')
    no_local_array=np.zeros(96).tolist()
    simcfgorig['sources']['LOCALS']['creation_rate'] = no_local_array  
    saving_file = args.citytag + '_no_locals' #str(list_delta_u_attraction[1]) is simply 0         
    if new_source_bool:
      if not os.path.exists(os.path.join(state_basename,layer_directory_to_add)):
        os.mkdir(os.path.join(state_basename,layer_directory_to_add))
    simcfgorig['state_basename'] = os.path.join(state_basename,layer_directory_to_add,saving_file)
    print('state_base_name:\t',simcfgorig['state_basename'])
#
  elif not locals_bool and att_act:
    no_local_array=np.zeros(96).tolist()  
    simcfgorig['sources']['LOCALS']['creation_rate'] = no_local_array  
    if  list_delta_u_attraction[1]>0:
      saving_file = args.citytag + list_delta_u_attraction[0] + str(list_delta_u_attraction[1])          
    else:
          saving_file = args.citytag + list_delta_u_attraction[0] +'_'+str(list_delta_u_attraction[1]).split('.')[1]
    if new_source_bool:
      if not os.path.exists(os.path.join(state_basename,layer_directory_to_add)):
        os.mkdir(os.path.join(state_basename,layer_directory_to_add))
    simcfgorig['state_basename'] = os.path.join(state_basename,layer_directory_to_add,saving_file)
#
  else:
    saving_file = args.citytag         
    if new_source_bool:
      if not os.path.exists(os.path.join(state_basename,layer_directory_to_add)):
        os.mkdir(os.path.join(state_basename,layer_directory_to_add))
    simcfgorig['state_basename'] = os.path.join(state_basename,layer_directory_to_add,saving_file)
  return True    

def dealing_sources(new_source_bool,name_source_list,df_avg,simcfgorig,locals_bool):    
  print('dealing sources')
  if locals_bool:
    no_local_array=np.zeros(96).tolist()
    simcfgorig['sources']['LOCALS']['creation_rate'] = no_local_array    
  if new_source_bool:
    name_source_list = args.new_source_list.split('-')
    for name_source in name_source_list: 
      print('add source',name_source)
      add_source(df_avg,simcfgorig,name_source)
#    print('il file di configurazione con sorgenti aggiunte è \n',simcfgorig['sources'])
  if reset_source_bool:
    reset_source_list = args.reset_source_list.split('-')
    print('resetto le sorgenti', reset_source_list)
    for r_source in reset_source_list: 
      reset_source(simcfgorig,r_source)
#    print('il file di configurazione è con sorgenti resettate \n',simcfgorig['sources'])

  f= False
  if f:
    temporary_sources= args.change_source_list.split('-')
    l_sources_to_set = []
    for s in temporary_sources:
      l_sources_to_set.append(s.upper())
    set_simcfg_fluxes_from_list_sources(df_avg,simcfgorig,l_sources_to_set) # default ['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN'] #['SCALZI_3_IN','SCALZI_2_IN']
  return True    

def add_attraction(simcfgorig):
  data_barriers = pd.read_csv(dir_data,';')      
  if args.add_attractions_bool:
    name_attractions = args.name_attractions.split('-')        
    for name_attraction in name_attractions:
          simcfgorig['attractions'][name_attraction] = {
          'lat' : data_barriers.loc[data_barriers['Description']==name_attraction.upper()+'_IN'].iloc[0]['Lat'],
          'lon' : data_barriers.loc[data_barriers['Description']==name_attraction.upper()+'_IN'].iloc[0]['Lon'],
          'weight': list(np.ones(24)*0.6),
          'timecap':list(np.ones(24)*1000),
          'visit_time': 2880   
          }
  print(simcfgorig['attractions'][name_attraction])
  return True

#Define function to be runned in parallel
def run_simulation(list_delta_u_attraction):
# initialize simcfgorig
  conf_dir='/home/aamad/codice/slides/work_slides/conf_files'
  with open(os.path.join(conf_dir,'conf_venezia.json')) as g:
    simcfgorig = json.load(g)
# Answering: new sources? I want locals? Do I change the weight of the attractions 
  new_source_bool = args.new_source_bool        
  if new_source_bool:
    name_source_list = args.new_source_list
  locals_bool = args.locals
  att_act =args.attraction_activate
# Define the sources
  dealing_sources(new_source_bool,name_source_list,df_avg,simcfgorig,locals_bool)
  layer_directory_to_add = sources_simulation_with_people(simcfgorig)
# setting saving dir
  add_attraction(simcfgorig)
  setting_state_base_name(att_act,locals_bool,new_source_bool,list_delta_u_attraction,simcfgorig,layer_directory_to_add)
# lancio la simulazione
  cfg = conf(simcfgorig0)
  simcfg = cfg.generate(start_date = args.start_date, stop_date = args.stop_date, citytag = args.citytag)#(override_weights=w)
# Adding piece where I change the sources instead of conf.py
  simcfgs = json.dumps(simcfgorig)
  s = simulation(simcfgs)
  print(s.sim_info())
  s.run()
  return True  


################################
# SETTING ENVIRONMENT VARIABLE #
################################
try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'bin'))
  from pysim import simulation

  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
  from conf import conf
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e
    
#print('directory simulation',dir(simulation))


#######################################
###### GLOBAL DIRECTORIES FOR CONF ###
#######################################

#conf_dir0 = r'C:\Users\aamad\phd_scripts\codice\slides\pvt\conf'
conf_dir0 = os.path.join(os.environ['WORKSPACE'],'slides','pvt','conf')
#conf_dir=r'C:\Users\aamad\phd_scripts\codice\slides\work_15_07'
conf_dir=os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files')
#dir_data=r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\data\barriers_config.csv'
dir_data = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data','barriers_config.csv')
#state_basename = r"C:\Users\aamad\phd_scripts\codice\slides\work_ws\output"
state_basename = os.path.join(os.environ['WORKSPACE'],'slides','work_ws','output')
#file_distances_real_data = r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\data\COVE flussi_pedonali 18-27 luglio.xlsx'
file_distances_real_data =os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data','COVE flussi_pedonali 18-27 luglio.xlsx')
real_data=pd.read_excel(file_distances_real_data, engine='openpyxl')
time_format='%Y-%m-%d %H:%M:%S'



###############################
#             MAIN            #
###############################
#import logging
#import coloredlogs

#console_formatter = coloredlogs.ColoredFormatter('%(asctime)s [%(levelname)s] (%(name)s:%(funcName)s) %(message)s', "%H:%M:%S")
#console_handler = logging.StreamHandler()
#console_handler.setFormatter(console_formatter)
#logging.basicConfig(
#  level=logging.DEBUG,
#  handlers=[console_handler]
#)
#logging.getLogger('matplotlib').setLevel(logging.WARNING)

# leggo il file di configurazione conf.json.local.albi.make
with open(os.path.join(conf_dir0,'conf.json.local.albi.make')) as g:
    simcfgorig0 = json.load(g)
#Leggo il json conf_venezia.json associato alla parte di simulazione in cui ho tirato già giù il numero di persone
with open(os.path.join(conf_dir,'conf_venezia.json')) as g:
    simcfgorig = json.load(g)
# define iterable for pool map 

if args.averaging_fluxes:
  df_avg = average_fluxes(real_data)
else:
  start_date=args.start_date 
  df_avg = pick_day_input(real_data,start_date)

list_delta_u_attraction = create_list_delta_u_attraction(args)
reset_source_list = args.reset_source_list
#Variables per multiprocessing
nsim = len(list_delta_u_attraction)
nagents = 6
chunksize = nsim // nagents
#INITIALIZING BOOL EXPRESSIONS
new_source_bool = args.new_source_bool
attraction_perturbation_activate = args.attraction_activate
locals_bool = args.locals
reset_source_bool = args.reset_source_bool
if attraction_perturbation_activate:
  tnow = datetime.now()
  # nagents numero di thread da utilizzare
  with Pool(processes=nagents) as pool:
    #sched_para.items() is the iterable of different simulations I have
    # len(results)=sched_para.items()
    # chunksize mi come dividere le simulazioni 
    result = pool.map(run_simulation, list_delta_u_attraction, chunksize)
  tscan = datetime.now() - tnow
  print(f'Scan took {tscan}')
else:
    print('no parallel needed, shape list_delta_u_attraction:\t',np.shape(list_delta_u_attraction))
    result = run_simulation(list_delta_u_attraction)

        
#controllare in /home/aamad/codice/covid_hep/python/scan/scan_par.py per idee su come gestire la parallelizzazione del programma
