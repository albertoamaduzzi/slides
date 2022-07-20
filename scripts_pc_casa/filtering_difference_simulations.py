import numpy as np
import pandas as pd
import json
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
from datetime import datetime
import time
##############################
##  SETTING CONFIGURATIONS  ##
##############################

parser=argparse.ArgumentParser(description='Insert starting and ending date of the simulation in form %Y-%m-%d %H:%M:%S default 2021-07-15')
parser.add_argument('--start_date',help='insert start date in format %Y-%m-%d %H:%M:%S, default:2021-07-15 00:00:00 ',type=str,default='2021-07-15 00:00:00')
parser.add_argument('--stop_date',help='insert start date in format %Y-%m-%d %H:%M:%S, default:2021-07-15 23:59:59',type=str,default='2021-07-15 23:59:59')
parser.add_argument('--citytag',help='insert citytag',type=str,default='venezia')
parser.add_argument('--dffile',help='insert file to be analized',type=str,default='/aamad/codice/slides/work_slides/data/venezia_barriers_210715_000000.csv')

args= parser.parse_args()

##########################
##########################
##########################

#####################
## GLOBAL VARIABLE ##
#####################
dffile = args.dffile #marzo 2022 Albi#
working_dir ='/aamad/codice/slides/work_slides/output'
conf_dir='/home/aamad/codice/slides/work_slides'
state_basename = "/home/aamad/codice/slides/work_ws/output"
time_format='%Y-%m-%d %H:%M:%S'
day_comparison = datetime.strptime(args.start_date, time_format)

# OPEN JSON FOR ATTRACTIONS
with open(os.path.join(conf_dir,'conf_files','conf_venezia.json')) as g:
    simcfgorig = json.load(g)
delta_u = [0.3, 0.25,0.15,0.1,0.05,-0.05,-0.1,-0.15, -0.2,-0.25,-0.3]
attractions = list(simcfgorig['attractions'].keys())# I am returned with the list of attractions
# SELEZIONO I FILE TRAMITE I PARAMETRI 
list_delta_u_attraction=[]
for attraction in attractions:
    for w in delta_u:
          list_delta_u_attraction.append([attraction,w])
          
          
tmp_str=args.start_date.replace('-','')
tmp_str=tmp_str.replace(':','')
time_title=tmp_str.replace(' ','_')          
time_title=time_title.replace('20','')



#####################
## MULTIPROCESSING ##
#####################
nagents = 30
chunksize =mp.cpu_count()
def creation_comparison(file_name,string_output,day_comparison):
    cest=2
    cest=0
    cet=1
    path3 = os.path.join('/home/aamad/codice/slides/work_slides/data/COVE flussi_pedonali 18-27 luglio.xlsx')
    path4 = file_name
    exc_flux=pd.read_excel(path3, engine='openpyxl')
    df_barriers=pd.read_csv(path4,';')
    working_dir='/home/aamad/codice/slides/work_slides/output'
    ### TAKING CARE JUST OF THE DAYS OF SIMULATION ###
    import datetime
    sulstice_summer=datetime.datetime(2021,3,21,0,0)
    sulstice_winter=datetime.datetime(2021,9,21,0,0)
    data_3=day_comparison

    if data_3.month>sulstice_summer.month and data_3.month<sulstice_winter.month: 
        exc_flux.timestamp=exc_flux.timestamp+datetime.timedelta(hours=cest)
    else:
        exc_flux.timestamp=exc_flux.timestamp+datetime.timedelta(hours=cet)
    flux_18=exc_flux.loc[exc_flux.timestamp>data_3]
    df_varco=flux_18.groupby(['varco']).sum()
    total_count_18_7=(df_varco['direzione']+df_varco['Unnamed: 3']).sum()
    ### RENDO COMPATIBILI I NOMI DELLE BARRIERE DELLA SIMULAZIONE CON QUELLE DEI DATI TELECAMERA ###
    list_barriers=[]
    for k in range(len(df_barriers.columns)):
        if (df_barriers.columns[k].find('_IN') != -1):
            barrier=(df_barriers.columns[k]).strip('_IN')
            list_barriers.append(barrier)
        elif (df_barriers.columns[k].find('_OUT') != -1):
            barrier=(df_barriers.columns[k]).strip('_OUT')
        else:
            continue
    #list_barriers=np.unique(df_barriers.columns)

    df_barriers=df_barriers[1:]
    time_=df_barriers.datetime
        ### PREPARO IL DATAFRAME ###
    dictionary_sim_data={'barrier':[],'time':[],'in_sim':[],'out_sim':[],'in+out_sim':[],'in_data':[],'out_data':[],'in+out_data':[]}
    df_comparison_sim_data=pd.DataFrame(dictionary_sim_data)
    dict_hour_data={}
    list_in_data=[]
    list_out_data=[]
    list_in_out_data=[]
    list_in_sim=[]
    list_out_sim=[]
    list_in_out_sim=[]
    list_time=[]
    list_barriere=[]
    count=0
    for hour in time_:    
        for barrier in list_barriers:
            temporary_flux=(flux_18.loc[flux_18.timestamp==hour])
            temporary_flux=temporary_flux.loc[temporary_flux.varco==barrier]
            if not temporary_flux.empty:
                tot_temporary_flux=(temporary_flux.iloc[0]['direzione']+temporary_flux.iloc[0]['Unnamed: 3'])
                list_in_data.append(temporary_flux.iloc[0]['direzione'])
                list_out_data.append(temporary_flux.iloc[0]['Unnamed: 3'])
                list_in_out_data.append(tot_temporary_flux)
            else:
                list_in_data.append(np.nan)
                list_out_data.append(np.nan)
                list_in_out_data.append(np.nan)
                print('the barrier {0} at time {1} does not exist'.format(barrier,hour))
                
            out_=df_barriers.iloc[count][barrier+'_OUT']
            in_=df_barriers.iloc[count][barrier+'_IN']
            in_out=df_barriers.iloc[count][barrier+'_OUT']+df_barriers.iloc[count][barrier+'_IN']
            list_in_sim.append(int(in_))
            list_out_sim.append(int(out_))
            list_in_out_sim.append(int(in_out))
            list_time.append(datetime.datetime.strptime(hour,'%Y-%m-%d %H:%M:%S'))
            list_barriere.append(barrier)
        count=count+1
    df_comparison_sim_data['barrier']=list_barriere
    df_comparison_sim_data['time']=pd.Series(list_time)-datetime.timedelta(hours=cest)
    df_comparison_sim_data['in_sim']=list_in_sim
    df_comparison_sim_data['out_sim']=list_out_sim
    df_comparison_sim_data['in+out_sim']=list_in_out_sim
    df_comparison_sim_data['in_data']=list_in_data
    df_comparison_sim_data['out_data']=list_out_data
    df_comparison_sim_data['in+out_data']=list_in_out_data
    df_comparison_sim_data.to_csv(os.path.join(working_dir,'df_comparison_sim_data_{}.csv'.format(string_output)),';')
    return True

def difference_matrix(list_delta_u_attraction):
    conf_dir = '/home/aamad/codice/slides/work_slides'
    if  list_delta_u_attraction[1]>0:
        string_parameters = args.citytag + list_delta_u_attraction[0] + str(list_delta_u_attraction[1]).split('.')[1]+'_barriers_'+time_title+'.csv'         
    else:
        string_parameters = args.citytag + list_delta_u_attraction[0] +'_'+str(list_delta_u_attraction[1]).split('.')[1]+'_barriers_'+time_title+'.csv'
    file_to_compare = os.path.join(state_basename,string_parameters)
    
    
    creation_comparison(file_to_compare,string_parameters,day_comparison)

    return file_to_compare,True  


go=True

if go:
    tnow = datetime.now()
  # nagents numero di thread da utilizzare
    with Pool(processes=nagents) as pool:
    #sched_para.items() is the iterable of different simulations I have
    # len(results)=sched_para.items()
    # chunksize mi come dividere le simulazioni 
        result = pool.map(difference_matrix, list_delta_u_attraction, chunksize)
    tscan = datetime.now() - tnow
    print(f'Time for creation dataframe comparison:    {tscan}')


    