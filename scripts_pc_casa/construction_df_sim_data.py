###IMPORTING FILE TO AVOID THAT EACH TIME EFFECTS ON THE DATAFRAME ARE ADDED and for example I find time_ starting at 7
import pandas as pd 
import os
import time
import numpy as np
start_global=time.time()
start=time.time()
data1=str(210718)
data2=str(210717)
data3=str(210716)
data3=str(210715)
number_people1=str(40)
number_people2=str(35)
number_people3=str(25)





#orario italiano rispetto UTC italiano di estate e inverno rispettivamente
cest=2
cest=0
cet=1
#env=os.environ()
#position=os.getcwd()
path1=os.path.join(r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/venezia_netstate_{}_000000.csv'.format(data3))
path2=os.path.join(r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/venezia_population_{}_000000.csv'.format(data3))
path3=os.path.join(r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/COVE flussi_pedonali 18-27 luglio.xlsx')
path4=r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output/venezia_barriers_{}_000000.csv'.format(data3)
#df_netstate=pd.read_csv(path1,';')
#df_population=pd.read_csv(path2,';')
exc_flux=pd.read_excel(path3, engine='openpyxl')
df_barriers=pd.read_csv(path4,';')
working_dir=r'C:/Users/aamad/phd_scripts/codice/slides/work_ws/output'
end=time.time()
print('time extraction data is {}'.format(end-start))
print('file scr_data {0}\nfile barriers from simulations {1}'.format(path3,path4))
### TAKING CARE JUST OF THE DAYS OF SIMULATION ###
import datetime
sulstice_summer=datetime.datetime(2021,3,21,0,0)
sulstice_winter=datetime.datetime(2021,9,21,0,0)
data_1=datetime.datetime(2021,7,18,0,0)
data_2=datetime.datetime(2021,7,17,0,0)
data_3=datetime.datetime(2021,7,16,0,0)
data_3=datetime.datetime(2021,7,15,0,0)
if data_3.month>sulstice_summer.month and data_3.month<sulstice_winter.month: 
#    print(exc_flux.timestamp)
    exc_flux.timestamp=exc_flux.timestamp+datetime.timedelta(hours=cest)
#    print(exc_flux.timestamp)
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
df_comparison_sim_data.to_csv(os.path.join(working_dir,'df_comparison_sim_data_{0}_{1}.csv'.format(number_people3,data3)),';')
end_global=time.time()
print('tempo totale di run programma = {}'.format(end_global-start_global))
df_comparison_sim_data
