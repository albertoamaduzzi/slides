import seaborn as sns
import matplotlib.pyplot as plt
data1=str(210718)
data2=str(210717)
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

##IN+OUT
for barrier in list_barriers:
    data=df_comparison_sim_data.groupby('barrier').get_group(barrier)
#    plt.subplot(len(df_comparison_sim_data.groupby('barrier')),1,1)
    sns.lineplot(data=data, x="time", y="in+out_sim",sizes=(15,10))
    plt.xticks(rotation=90)
    sns.lineplot(data=data, x="time", y="in+out_data",sizes=(15,10))
    plt.xticks(rotation=90)
    plt.title('{}'.format(barrier))
    plt.legend(['in+out_sim','in+out_data'])
    plt.show()
    

##OUT


for barrier in list_barriers:
    data=df_comparison_sim_data.groupby('barrier').get_group(barrier)
#    plt.subplot(len(df_comparison_sim_data.groupby('barrier')),1,1)
    sns.lineplot(data=data, x="time", y="out_sim",sizes=(15,10))
    plt.xticks(rotation=90)
    sns.lineplot(data=data, x="time", y="out_data",sizes=(15,10))
    plt.xticks(rotation=90)
    plt.title('{}'.format(barrier))
    plt.legend(['out_sim','out_data'])
    plt.show()
    


##IN
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
