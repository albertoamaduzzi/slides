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
parser.add_argument('--start_date',help='insert start date in format %Y-%m-%d %H:%M:%S, default:2021-07-15 00:00:00 ',type=str,default='2021-07-15 00:00:00')
parser.add_argument('--stop_date',help='insert start date in format %Y-%m-%d %H:%M:%S, default:2021-07-15 23:59:59',type=str,default='2021-07-15 23:59:59')
parser.add_argument('--citytag',help='insert citytag',type=str,default='venezia')
parser.add_argument('-a','--attraction_activate',help='insert bool to activate changes in the attractions',type=bool,default=False)
parser.add_argument('-l', '--locals',help='insert locals true in the case of no_locals',type=bool,default=True)
parser.add_argument('-ld','--local_distribution',help='insert distribution',type=str,default='none')
parser.add_argument('-af','--averaging_fluxes',help='True if I want to average sources over the week',type=bool,default=False)

args= parser.parse_args()

#%%
def pick_day_input(df,start_date):
  time_format='%Y-%m-%d %H:%M:%S'
  ar_varc=np.unique(np.array(df['varco'],dtype=str))
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
  return df_new

#%%  
def average_fluxes(df):
  ar_varc=np.unique(np.array(df['varco'],dtype=str))
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
  h_mask=np.arange(25)
  tags = list(df.columns.drop('datetime'))
  tagn=len(tags)
  df_avg = pd.DataFrame(data=np.zeros([25,tagn+1]),columns=df.columns)
#  df_avg.set_index(keys=h_mask)
#  print('mask and index',df_avg.index)
  week_mask = [True if h.weekday()<4 else False for h in df.datetime]
  dfc_temp=df.loc[week_mask]
  for e in h_mask:
    mask=[True if h.hour==e else False for h in dfc_temp.datetime]                                          
    df_avg.iloc[e][tags]=dfc_temp[tags].loc[mask].mean()
    df_avg.iloc[e]['datetime']=e

  return df_avg

def extract_sources_fluxes(df_avg,simcfgorig,list_name_sources):
  '''Mi sto riferendo nel caso d'uso a questa lista in input ['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN','SCALZI_1_IN']'''
  if args.averaging_fluxes:
    df_avg =df_avg[:-1]
  else:
    pass
  list_fluxes_conf_file = []
  for name_source in list_name_sources:
    list_flux_single_source = []
    for flux in df_avg[name_source]:
      for i in range(4):
        list_flux_single_source.append(flux/4)
    if len(list_flux_single_source)!= 96:
      print('la lunghezza della lista per la sorgente {0} è {1}'.format(name_source,len(list_flux_single_source)))
      exit( )
    list_fluxes_conf_file.append(list_flux_single_source)
    name_source = name_source.split('_')[0]+'_'+ name_source.split('_')[2]
    new_source_name = ''
    for position_letter in range(len(name_source)):
      if position_letter == 0 or position_letter ==len(name_source)-1 or position_letter ==len(name_source)-2:
        new_source_name = new_source_name + name_source[position_letter]
      else:
        new_source_name = new_source_name + name_source[position_letter].lower() 
    simcfgorig['sources'][new_source_name]['creation_rate'] = list_flux_single_source
    return simcfgorig

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



#Define function to be runned in parallel
def run_simulation(list_delta_u_attraction):
#  conf_dir = r'C:\Users\aamad\phd_scripts\codice\slides\work_15_07'        
  conf_dir='/home/aamad/code/slides/work_15_07'
  with open(os.path.join(conf_dir,'conf_venezia.json')) as g:
    simcfgorig = json.load(g)
  locals_bool = args.locals
  att_act =args.attraction_activate
  if att_act:
    simcfgorig['attractions'][attraction]['weight'] = list(np.array(simcfgorig['attractions'][attraction]['weight']) + np.ones(len(np.array(simcfgorig['attractions'][attraction]['weight'])))*w)
#HANDLING NO LOCALS
  if locals_bool and att_act:
    no_local_array=np.zeros(96).tolist()
    simcfgorig['sources']['LOCALS']['creation_rate'] = no_local_array  
#
    if  list_delta_u_attraction[1]>0:
      saving_file = args.citytag + list_delta_u_attraction[0] + str(list_delta_u_attraction[1]).split('.')[1]+'_no_locals'         
    else:
          saving_file = args.citytag + list_delta_u_attraction[0] +'_'+str(list_delta_u_attraction[1]).split('.')[1]+'_no_locals'
    simcfgorig['state_basename'] = os.path.join(state_basename,saving_file)
#
  elif locals_bool and not att_act:
    print('I am in the case no locals and no variation of attraction')
    no_local_array=np.zeros(96).tolist()
    simcfgorig['sources']['LOCALS']['creation_rate'] = no_local_array  
    saving_file = args.citytag + '_no_locals' #str(list_delta_u_attraction[1]) is simply 0         
    simcfgorig['state_basename'] = os.path.join(state_basename,saving_file)
    print('state_base_name:\t',simcfgorig['state_basename'])
#
  elif not locals_bool and att_act:
    no_local_array=np.zeros(96).tolist()  
    simcfgorig['sources']['LOCALS']['creation_rate'] = no_local_array  
    if  list_delta_u_attraction[1]>0:
      saving_file = args.citytag + list_delta_u_attraction[0] + str(list_delta_u_attraction[1])          
    else:
          saving_file = args.citytag + list_delta_u_attraction[0] +'_'+str(list_delta_u_attraction[1]).split('.')[1]
    simcfgorig['state_basename'] = os.path.join(state_basename,saving_file)
#
  else:
    saving_file = args.citytag         
    simcfgorig['state_basename'] = os.path.join(state_basename,saving_file)
# lancio la simulazione
  cfg = conf(simcfgorig0)
  simcfg = cfg.generate(start_date = args.start_date, stop_date = args.stop_date, citytag = args.citytag)#(override_weights=w)
# Adding piece where I change the sources instead of conf.py
  extract_sources_fluxes(df_avg,simcfgorig,['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN'])
  simcfgs = json.dumps(simcfgorig)
  print(simcfgs)
  s = simulation(simcfgs)
  print(s.sim_info())
  s.run()
  return saving_file,True  


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
    
print('directory simulation',dir(simulation))


#######################################
###### GLOBAL DIRECTORIES FOR CONF ###
#######################################

#conf_dir0 = r'C:\Users\aamad\phd_scripts\codice\slides\pvt\conf'
conf_dir0 = '/home/aamad/code/slides/pvt/conf'
#conf_dir=r'C:\Users\aamad\phd_scripts\codice\slides\work_15_07'
conf_dir='/home/aamad/code/slides/work_15_07'
#dir_data=r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\data\barriers_config.csv'
dir_data = '/home/aamad/code/slides/work_ws/data/barriers_config.csv'
#state_basename = r"C:\Users\aamad\phd_scripts\codice\slides\work_ws\output"
state_basename = "/home/aamad/code/slides/work_ws/output"
#file_distances_real_data = r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\data\COVE flussi_pedonali 18-27 luglio.xlsx'
file_distances_real_data ='/home/aamad/code/slides/work_ws/data/COVE flussi_pedonali 18-27 luglio.xlsx'
real_data=pd.read_excel(file_distances_real_data, engine='openpyxl')
time_format='%Y-%m-%d %H:%M:%S'



###############################
#             MAIN            #
###############################

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
#Variables per multiprocessing
nsim = len(list_delta_u_attraction)
nagents = 6
chunksize = nsim // nagents
attraction_perturbation_activate = args.attraction_activate
locals_bool = args.locals
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
