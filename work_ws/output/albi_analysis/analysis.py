###IMPORTING FILE TO AVOID THAT EACH TIME EFFECTS ON THE DATAFRAME ARE ADDED and for example I find time_ starting at 7

###Il mio scopo è analizzare giorno per giorno separatamente i flussi che misuriamo con le telecamere con i flussi che simuliamo.
import pandas as pd 
import os
import time
import numpy as np
###Naming the tests and their corrispective paths###
start_global=time.time()
start=time.time()
data1=str(210718)
data2=str(210717)
data3=str(210716)
number_people1=str(40)
number_people2=str(35)
number_people3=str(25)
#orario italiano rispetto UTC italiano di estate e inverno rispettivamente
cest=2
cet=1
path1=(r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/venezia_netstate_{}_000000.csv'.format(data3))
path2=(r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/venezia_population_{}_000000.csv'.format(data3))
path3=(r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/COVE flussi_pedonali 18-27 luglio.xlsx')
path4=r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/venezia_barriers_{}_000000.csv'.format(data3)
#df_netstate=pd.read_csv(path1,';')
#df_population=pd.read_csv(path2,';')
exc_flux=pd.read_excel(path3, engine='openpyxl')
df_barriers=pd.read_csv(path4,';')
working_dir=r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output'
end=time.time()
print('time extraction data is {}'.format(end-start))

### TAKING CARE JUST OF THE DAYS OF SIMULATION ###
import datetime
data_1=datetime.datetime(2021,7,18,0,0)
data_2=datetime.datetime(2021,7,17,0,0)
data_3=datetime.datetime(2021,7,16,0,0)
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
#### LA PRIMA RIGA DI DF_BARRIERS è INVALIDA
df_barriers=df_barriers[1:]

time_=df_barriers.datetime
print(time_)
count=0
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
        list_time.append(hour)
        list_barriere.append(barrier)
    count=count+1
df_comparison_sim_data['barrier']=list_barriere
df_comparison_sim_data['time']=list_time
df_comparison_sim_data['in_sim']=list_in_sim
df_comparison_sim_data['out_sim']=list_out_sim
df_comparison_sim_data['in+out_sim']=list_in_out_sim
df_comparison_sim_data['in_data']=list_in_data
df_comparison_sim_data['out_data']=list_out_data
df_comparison_sim_data['in+out_data']=list_in_out_data
df_comparison_sim_data.to_csv(os.path.join(working_dir,'df_comparison_sim_data_{0}_{1}.csv'.format(number_people3,data3)),';')
end_global=time.time()
print('tempo totale di run programma = {}'.format(end_global-start_global))
df_comparison_sim_data
#### df_comparison_sim_data è il dataframe che utilizzo per mettere i dati confrontati.


#### Ora faccio i plot: for source in list_sources xlabel(time),ylabel(numberofpeople)

import seaborn as sns
import matplotlib.pyplot as plt
path4=r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/venezia_barriers_{}_000000.csv'.format(data3)
df_barriers=pd.read_csv(path4,';')
list_barriers=[]
for k in range(len(df_barriers.columns)):
    if (df_barriers.columns[k].find('_IN') != -1):
        barrier=(df_barriers.columns[k]).strip('_IN')
        list_barriers.append(barrier)
    elif (df_barriers.columns[k].find('_OUT') != -1):
        barrier=(df_barriers.columns[k]).strip('_OUT')
    else:
        continue

working_dir=r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output'
df_comparison_sim_data=pd.read_csv(os.path.join(working_dir,'df_comparison_sim_data_{}_{}.csv'.format(number_people3,data3)),';')
#fig, axes = plt.subplots(len(list_barriers),1)
#fig.suptitle('comparison \'_in\' simulation data all the barriers')
for barrier in list_barriers:
    data=df_comparison_sim_data.groupby('barrier').get_group(barrier)
#    plt.subplot(len(df_comparison_sim_data.groupby('barrier')),1,1)
    sns.lineplot(data=data, x="time", y="in_sim",sizes=(15,10))
    plt.xticks(rotation=90)
    sns.lineplot(data=data, x="time", y="in_data",sizes=(15,10))
    plt.xticks(rotation=90)
    plt.title('{}'.format(barrier))
    plt.legend(['in_sim','in_data'])
    plt.show()
    