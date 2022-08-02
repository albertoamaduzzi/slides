import argparse 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
'''  In input I have the excel file with people IN and OUT from each barrier (53). It is the one averaged for the week. The other input is 
the file barriers which venezia_(no_locals)_barriers_(day).csv'''

parser=argparse.ArgumentParser(description='Insert the simulation of interest to compare with the real data')
parser.add_argument('--date',help='insert date of simulation, default:210715 ',type=str,default='210715')
parser.add_argument('-n','--no_locals',help='insert number of locals',type=str,default='no_locals')
args= parser.parse_args()

#GLOBAL VARIABLES
date = args.date
number_locals = args.no_locals

#INPUT
file_distances_real_data = r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\data\COVE flussi_pedonali 18-27 luglio.xlsx'
file_distances_simulation = r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\data\venezia_no_locals_barriers_{}_000000.csv'.format(date)
exc_real_data = pd.read_excel(file_distances_real_data,engine='openpyxl')
df_simulation = pd.read_csv(file_distances_simulation,';')

#os.path.join(saving_dir,'df_simulation_dist_{}.csv'.format(day.strftime('%d-%m-%Y')))
list_strada_nova=['COSTITUZIONE_1','PAPADOPOLI_1','SCALZI_1','SCALZI_2','SCALZI_3','FARSETTI_1','MADDALENA_1','SANFELICE_1','SANTASOFIA_1','PISTOR_1','APOSTOLI_1','GRISOSTOMO_2','SALVADOR_4']

###IMPORTING FILE TO AVOID THAT EACH TIME EFFECTS ON THE DATAFRAME ARE ADDED and for example I find time_ starting at 7
start_global=time.time()
start=time.time()
#orario italiano rispetto UTC italiano di estate e inverno rispettivamente
cest=2
cest=0
cet=1

if not os.path.exists(r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/sim_real_comparison'):
    os.mkdir(r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/sim_real_comparison')
    working_dir=r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/sim_real_comparison'
else:
    working_dir=r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/sim_real_comparison'

end=time.time()
print('time extraction data is {}'.format(end-start))
### TAKING CARE JUST OF THE DAYS OF SIMULATION ###
import datetime
sulstice_summer=datetime.datetime(2021,3,21,0,0)
sulstice_winter=datetime.datetime(2021,9,21,0,0)
data_1=datetime.datetime(2021,7,18,0,0)
data_2=datetime.datetime(2021,7,17,0,0)
data_3=datetime.datetime(2021,7,16,0,0)
data_3=datetime.datetime(2021,7,15,0,0)
if data_3.month>sulstice_summer.month and data_3.month<sulstice_winter.month: 
#    print(exc_real_data.timestamp)
    exc_real_data.timestamp=exc_real_data.timestamp+datetime.timedelta(hours=cest)
#    print(exc_real_data.timestamp)
else:
    exc_real_data.timestamp=exc_real_data.timestamp+datetime.timedelta(hours=cet)

flux_18=exc_real_data.loc[exc_real_data.timestamp>data_3]
df_varco=flux_18.groupby(['varco']).sum()
total_count_18_7=(df_varco['direzione']+df_varco['Unnamed: 3']).sum()


### RENDO COMPATIBILI I NOMI DELLE BARRIERE DELLA SIMULAZIONE CON QUELLE DEI DATI TELECAMERA ###
#list_barriers=[]
#for k in range(len(df_simulation.columns)):
#    if (df_simulation.columns[k].find('_IN') != -1):
#        barrier=(df_simulation.columns[k]).strip('_IN')
#        list_barriers.append(barrier)
#    elif (df_simulation.columns[k].find('_OUT') != -1):
#        barrier=(df_simulation.columns[k]).strip('_OUT')
#    else:
#        continue
#list_barriers=np.unique(df_simulation.columns)

df_simulation=df_simulation[1:]
#df_simulation.reset_index(inplace=True)
#df_simulation = df_simulation.rename(columns = {'index':'datetime'})
time_=df_simulation['timestamp']
from datetime import datetime
time_=[datetime.fromtimestamp(t) for t in time_]
count=0
### PREPARO IL DATAFRAME ###
list_barriers = list_strada_nova
print('list barriers',list_barriers)
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
            
        out_=df_simulation.iloc[count][barrier+'_OUT']
        in_=df_simulation.iloc[count][barrier+'_IN']
        in_out=df_simulation.iloc[count][barrier+'_OUT']+df_simulation.iloc[count][barrier+'_IN']
        list_in_sim.append(int(in_))
        list_out_sim.append(int(out_))
        list_in_out_sim.append(int(in_out))
        list_time.append(hour)
        list_barriere.append(barrier)
    count=count+1
df_comparison_sim_data['barrier']=list_barriere
from datetime import timedelta
print(pd.Series(list_time))
df_comparison_sim_data['time']=pd.Series(list_time,dtype='datetime64[s]')-timedelta(hours=cest)
df_comparison_sim_data['in_sim']=list_in_sim
df_comparison_sim_data['out_sim']=list_out_sim
df_comparison_sim_data['in+out_sim']=list_in_out_sim
df_comparison_sim_data['in_data']=list_in_data
df_comparison_sim_data['out_data']=list_out_data
df_comparison_sim_data['in+out_data']=list_in_out_data

df_comparison_sim_data.to_csv(os.path.join(working_dir,'df_comparison_sim_data_{0}_{1}.csv'.format(number_locals,date)),';')
end_global=time.time()
print('tempo totale di run programma = {}'.format(end_global-start_global))

for barrier in list_barriers:
    data=df_comparison_sim_data.groupby('barrier').get_group(barrier)
    sns.lineplot(data=data, x="time", y="in_sim",sizes=(15,10))
    plt.xticks(rotation=90)
    sns.lineplot(data=data, x="time", y="in_data",sizes=(15,10))
    plt.xticks(rotation=90)
    plt.title('{}'.format(barrier))
    plt.legend(['in_sim','in_data'])
    plt.savefig(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\plots\confronto_sim_real_strada_nova\confronto_sim_data_in{}'.format(barrier))
    plt.show()
    