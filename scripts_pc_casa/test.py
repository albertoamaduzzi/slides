#! /usr/bin/env python3
import sys
import os
import json
import argparse
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from datetime import datetime
###########example bash command##############
# cd ../slides/scripts
# python3 test.py -a True -l True
# In this way in the list of change of parameters, that define the number of threads I will have no locals and change of attractions
#
### NEW VERSION IMPROVED ../scripts_pc_casa/sim_parallel.py

#Enter parameters start_,end_ date of simulation
parser=argparse.ArgumentParser(description='Insert starting and ending date of the simulation in form %Y-%m-%d %H:%M:%S default 2021-07-15')
parser.add_argument('--start_date',help='insert start date in format %Y-%m-%d %H:%M:%S, default:2021-07-15 00:00:00 ',type=str,default='2021-07-15 00:00:00')
parser.add_argument('--stop_date',help='insert start date in format %Y-%m-%d %H:%M:%S, default:2021-07-15 23:59:59',type=str,default='2021-07-15 23:59:59')
parser.add_argument('--citytag',help='insert citytag',type=str,default='venezia')
parser.add_argument('-a','--attraction_activate',help='insert bool to activate changes in the attractions',type=bool,default=False)
parser.add_argument('-l', '--locals',help='insert locals',type=bool,default=False)
parser.add_argument('-ld','--local_distribution',help='insert distribution',type=str,default='none')

args= parser.parse_args()

# Definition directory of conf,
conf_dir0 = '/home/aamad/codice/slides/pvt/conf'
conf_dir='/home/aamad/codice/slides/work_slides/conf_files'
dir_data='/home/aamad/codice/slides/work_slides/data/barriers_config.csv'
state_basename = "/home/aamad/codice/slides/work_ws/output"
#dffile = '/aamad/codice/slides/work_slides/data/venezia_barriers_210715_000000.csv' #marzo 2022 Albi#
working_dir ='/aamad/codice/slides/work_slides/output'
time_format='%Y-%m-%d %H:%M:%S'

# SETTING ENVIRONMENT VARIABLE
try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'bin'))
  from pysim import simulation

  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
  from conf import conf
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e


#Define function to be runned in parallel
def run_simulation(list_delta_u_attraction):
  conf_dir = '/home/aamad/codice/slides/work_slides'        
  with open(os.path.join(conf_dir,'conf_files','conf_venezia.json')) as g:
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
  simcfgs = json.dumps(simcfgorig)
  s = simulation(simcfgs)
  print(s.sim_info())
  s.run()
  return saving_file,True  


# leggo il file di configurazione conf.json.local.albi.make
with open(os.path.join(conf_dir0,'conf.json.local.albi.make')) as g:
    simcfgorig0 = json.load(g)
#Leggo il json conf_venezia.json associato alla parte di simulazione in cui ho tirato già giù il numero di persone
with open(os.path.join(conf_dir,'conf_files','conf_venezia.json')) as g:
    simcfgorig = json.load(g)
# define iterable for pool map 

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
#Variables per multiprocessing
nsim = len(list_delta_u_attraction)
nagents = 6
chunksize = nsim // nagents

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

'''

        with open(os.path.join(conf_dir,'conf_files','conf_venezia.json')) as g:
          simcfgorig = json.load(g)
        simcfgorig['attractions'][attraction]['weight'] = list(np.array(simcfgorig['attractions'][attraction]['weight']) + np.ones(len(np.array(simcfgorig['attractions'][attraction]['weight'])))*w)
        saving_file = args.citytag+attraction+str(w).split('.')[1]         
        simcfgorig['state_basename'] = os.path.join(state_basename,saving_file)
        print(simcfgorig['state_basename'])
# lancio la simulazione        
        cfg = conf(simcfgorig0)
        simcfg = cfg.generate(start_date = args.start_date, stop_date = args.stop_date, citytag = args.citytag)#(override_weights=w)
        simcfgs = json.dumps(simcfgorig)
        with open('check.json', 'w') as cout:
          json.dump(simcfgorig, cout, indent=2)
        s = simulation(simcfgs)
        print(s.sim_info())
        s.run()
'''        
        
#controllare in /home/aamad/codice/covid_hep/python/scan/scan_par.py per idee su come gestire la parallelizzazione del programma
