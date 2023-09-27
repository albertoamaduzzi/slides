import sys 
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
############################################ REPRODUCE NUMBER LOCALS COSTITUZIONE PAPADOPOLI 
lista_locals = []

for i in [13165  ,16479  ,14592  ,12331  ,10369  ,10058  ,11486  ,17261  ,34596  ,58961 ,83676 ,116294 ,153646 ,189855 ,221815 ,254315 ,287078 ,315350 ,344479 ,371515 ,393474 ,404686 ,410937 ,414293]:
  for j in range(4):
    lista_locals.append(i/4)
print(lista_locals)



############################################ PLOT RAPIDI SIM REALE
if 1==0:
  date = '2021-07-17 23:00:00'
  legend_ = ['sim','real']
  pt_ = os.path.join(os.environ['WORKSPACE'],'slides/work_slides/src_Scalzi_2_IN-Scalzi_3_IN-Papadopoli_1_IN---attr_Farsetti_1_IN-2021-07-17 23-00-00/output_sim_0')
  pt_1 = os.path.join(os.environ['WORKSPACE'],'slides/work_slides/src_Scalzi_2_IN---attr_Farsetti_1_IN-2021-07-17 23-00-00/output_sim_0')
  pt_2 = os.path.join(os.environ['WORKSPACE'],'slides/work_slides/src_Scalzi_2_IN-Scalzi_3_IN---attr_Farsetti_1_IN-2021-07-17 23-00-00/output_sim_0')
  pt_3 = os.path.join(os.environ['WORKSPACE'],'slides/work_slides/src_Scalzi_3_IN---attr_Farsetti_1_IN-2021-07-17 23-00-00/output_sim_0')
  df = pd.read_csv(pt_+'/venezia_barriers_210717_230000.csv',';')
  df1 = pd.read_csv(pt_1+'/venezia_barriers_210717_230000.csv',';')
  df2 = pd.read_csv(pt_2+'/venezia_barriers_210717_230000.csv',';')
  df3 = pd.read_csv(pt_3+'/venezia_barriers_210717_230000.csv',';')
  ptt = [pt_,pt_1,pt_2,pt_3] #lista path simulazione
  dff = [df,df1,df2,df3] # lista df_simulazione
  list_dir = ['SC2_SC3_PP','SC2','SC2_SC3','SC3'] # lista_output_sorgenti_sim
  list_date = ['2021-07-17 23:00:00','2021-07-17 23:00:00','2021-07-17 23:00:00','2021-07-17 23:00:00']
  list_df_day = [pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}.csv'.format(list_date[k].split(' ')[0]),';') for k in range(len(list_date))]
  #df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}.csv'.format(date.split(' ')[0]),';')
  list_cols = ['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN','SCALZI_2_IN','SCALZI_3_IN']
  h=np.arange(23)
  for i in range(len(ptt)):
    #h=list_df_day[i]['timestamp']
    for col in list(list_cols):
      plt.plot(h,dff[i][col][1:])
      plt.plot(h,list_df_day[i][col][:-1])
      plt.xlabel('time')
      plt.ylabel('flux')
      plt.title(col + ' ' + list_date[i])
      plt.legend(legend_)
      plt.xticks(rotation = 30)
      if not os.path.exists(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')+'/{}'.format(list_dir[i])):
        os.mkdir(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')+'/{}'.format(list_dir[i]))
      plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')+'/{}'.format(list_dir[i])+'/sim_real_{}_{}.png'.format(col,list_date[i].split(' ')[0]), dpi = 150)
      plt.show()
#############################################

############################### INVARIANCE FOR TRANSLATION OF TIME #############################
if 1==0:
  start_date =  ['2021-07-12 23-00-00','2021-07-13 23-00-00','2021-07-14 23-00-00','2021-07-15 23-00-00']#,'2021-07-16 23-00-00','2021-07-17 23-00-00']
  fig = plt.figure()
  h=np.arange(22)
  for date in start_date:
    df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}.csv'.format(date.split(' ')[0]),';')
    plt.plot(h,df_day['SCALZI_2_IN'][2:])
  plt.title('Scalzi 2 week')
  plt.legend(start_date)
  if not os.path.exists(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti')):
    os.mkdir(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti'))
  plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti')+ '/SCALZI_2.png',dpi = 250)
  plt.show()

  fig = plt.figure()
  h=np.arange(22)
  for date in start_date:
    df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}.csv'.format(date.split(' ')[0]),';')
    plt.plot(h,df_day['SCALZI_3_IN'][2:])
  plt.title('Scalzi 3 week')
  plt.legend(start_date)
  if not os.path.exists(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti')):
    os.mkdir(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti'))
  plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti')+ '/SCALZI_3.png',dpi = 250)
  plt.show()

  fig = plt.figure()
  h=np.arange(22)
  for date in start_date:
    df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}.csv'.format(date.split(' ')[0]),';')
    plt.plot(h,df_day['PAPADOPOLI_1_IN'][2:])
  plt.title('Papadopoli week')
  plt.legend(start_date)
  if not os.path.exists(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti')):
    os.mkdir(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti'))
  plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti')+ '/PAPADOPOLI.png',dpi = 250)
  plt.show()

  fig = plt.figure()
  h=np.arange(22)
  for date in start_date:
    df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}.csv'.format(date.split(' ')[0]),';')
    plt.plot(h,df_day['COSTITUZIONE_1_IN'][2:])
  plt.title('Costituzione week')
  plt.legend(start_date)
  if not os.path.exists(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti')):
    os.mkdir(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti'))
  plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot','settimanale_singole_sorgenti')+ '/COSTITUZIONE.png',dpi = 250)
  plt.show()



#################################################################################################
start_date =  ['2021-07-12 23-00-00']#,'2021-07-13 23-00-00','2021-07-14 23-00-00','2021-07-15 23-00-00']#,'2021-07-16 23-00-00','2021-07-17 23-00-00']
#['2021-07-17 23-00-00']


if 1==1:
  fig,axs = plt.subplots(1,1,sharey = True,figsize = (15,10))
  c = 0
  for date in start_date:
    df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}.csv'.format(date.split(' ')[0]),';')
    corrMatrix = df_day[['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN','SCALZI_2_IN','SCALZI_3_IN']].corr()
    if c<5:
      a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax = axs)#axs[c])
      a.set_title(date)
    else:
      a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax = axs)#axs[c])
      a.set_title(date)
    c = c + 1      
  fig.suptitle('correlation SORGENTI')
    
    
  if not os.path.exists(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')):
    os.mkdir(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot'))
  plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')+'/SORGENTI_correlation.png',dpi = 250)# '/SORGENTI_correlation_week.png')
  plt.show()

  fig,axs = plt.subplots(1,1,sharey = True,figsize = (15,10))
  c = 0
  for date in start_date:
    df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}.csv'.format(date.split(' ')[0]),';')
    corrMatrix = df_day[['SCALZI_2_IN','SCALZI_3_IN','MADDALENA_1_IN','FARSETTI_1_IN','SANFELICE_1_IN','SANTASOFIA_1_IN','APOSTOLI_1_IN']].corr() #'PISTOR_1_IN','GRISOSTOMO_1_IN']
    if c<5:
      a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax = axs)#axs[c])
      a.set_title(date)
    else:
      a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax = axs)#axs[c])
      a.set_title(date)
    c = c + 1   
  plt.yticks(rotation = 30)  
  fig.suptitle('correlation STRADA NOVA')

  plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')+'/STRADA_NOVA_correlation.png',dpi = 250)#+ '/STRADANOVA_correlation_week.png')
  plt.show()  




  fig,axs = plt.subplots(1,1,sharey = True,figsize = (15,10))
  c = 0
  for date in start_date:
    df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}.csv'.format(date.split(' ')[0]),';')
    corrMatrix = df_day[['PAPADOPOLI_1_IN','TREPONTI_1_IN','RAGUSEI_1_IN','RAGUSEI_2_IN','RAGUSEI_3_IN']].corr() #'COSTITUZIONE_1_IN','BARNABA_1_IN','CASINNOBILI_1_IN','RAGUSEI_4_IN'
    if c<5:
      a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax =axs)# axs[c])
      a.set_title(date)
    else:
      a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax = axs)#axs[c])
      a.set_title(date)
    c = c + 1      
  plt.yticks(rotation = 30)  

  fig.suptitle('correlation STRADA OCCIDENTALE')
  plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')+'/STRADAPAPADOPOLI_correlation.png',dpi =250)#+ '/STRADAPAPADOPOLI_correlation_week.png')
  plt.show()

  fig,axs = plt.subplots(1,1,sharey = True,figsize = (15,10))
  c = 0
  for date in start_date:
    df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data')+'/dataframe_real_data_pick_day_{}.csv'.format(date.split(' ')[0]),';')
    corrMatrix = df_day[['SCALZI_2_IN','SCALZI_3_IN','SANGIACOMO_1_IN','SANAGOSTIN_1_IN','TERAANTONIO_1_IN','MADONETA_1_IN']].corr() #,'MANDOLA_2_IN'
    if c<5:
      a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax = axs)#axs[c])
      a.set_title(date)
    else:
      a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax = axs)#axs[c])
      a.set_title(date)
    c = c + 1      
  plt.yticks(rotation = 30)    
  fig.suptitle('correlation VIA CENTRALE verso S.M.')
  plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')+ '/VIACENTRALE_correlation.png',dpi = 250)#+ '/VIACENTRALE_correlation_week.png')
  plt.show()


## CORRISPETTIVA ANALISI DELLE CORRELAZIONI NELLE SIMULAZIONI

list_new_sources = [["Scalzi_2_IN"],["Scalzi_2_IN","Scalzi_3_IN"],["Scalzi_2_IN","Scalzi_3_IN","Papadopoli_1_IN"],["Scalzi_2_IN","Scalzi_3_IN","Costituzione_1_IN"],["Scalzi_2_IN","Scalzi_3_IN","Costituzione_1_IN","Papadopoli_1_IN"],
                                                      ["Scalzi_2_IN","Papadopoli_1_IN"],["Scalzi_2_IN","Costituzione_1_IN"],["Scalzi_2_IN","Papadopoli_1_IN","Costituzione_1_IN"],
                                                      ["Scalzi_3_IN"],
                                                      ["Scalzi_3_IN","Papadopoli_1_IN"],["Scalzi_3_IN","Costituzione_1_IN"],["Scalzi_3_IN","Papadopoli_1_IN","Costituzione_1_IN"],
                                                      ["Papadopoli_1_IN"],
                                                      ["Papadopoli_1_IN","Costituzione_1_IN"],
                                                      ["Costituzione_1_IN"]]


fig,axs = plt.subplots(1,6,sharey = True,figsize = (30,20))
c = 0
for date in start_date:
#  df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','src_Scalzi_2_IN-Scalzi_3_IN-Costituzione_1_IN-Papadopoli_1_IN---attr_Farsetti_1_IN-{}'.format(date),'output_sim_0')+'/venezia_barriers_{0}_{1}.csv'.format(date.split(' ')[0].split('-')[0][2:]+date.split(' ')[0].split('-')[1]+date.split(' ')[0].split('-')[2],date.split(' ')[1].split('-')[0]+date.split(' ')[1].split('-')[1]+date.split(' ')[1].split('-')[2]),';')
  df_day = pd.read_csv(pt_+'/venezia_barriers_210717_230000.csv',';')

  corrMatrix = df_day[['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN','SCALZI_2_IN','SCALZI_3_IN']].corr()
  if c<5:
    a = sns.heatmap(corrMatrix ,annot = True,cbar = False,ax = axs[c])
    a.set_title(date)
  else:
    a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax = axs[c])
    a.set_title(date)
  c = c + 1      
fig.suptitle('correlation SORGENTI')

if not os.path.exists(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')):
      os.mkdir(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot'))
plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')+ '/SORGENTI_correlation_sim.png')
plt.show()

fig,axs = plt.subplots(1,6,sharey = True,figsize = (30,20))
c = 0
for date in start_date:
#  df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','src_Scalzi_2_IN-Scalzi_3_IN-Costituzione_1_IN-Papadopoli_1_IN---attr_Farsetti_1_IN-{}'.format(date),'output_sim_0')+'/venezia_barriers_{0}_{1}.csv'.format(date.split(' ')[0].split('-')[0][2:]+date.split(' ')[0].split('-')[1]+date.split(' ')[0].split('-')[2],date.split(' ')[1].split('-')[0]+date.split(' ')[1].split('-')[1]+date.split(' ')[1].split('-')[2]),';')
  df_day = pd.read_csv(pt_+'/venezia_barriers_210717_230000.csv',';')

  corrMatrix = df_day[['SCALZI_2_IN','SCALZI_3_IN','MADDALENA_1_IN','FARSETTI_1_IN','SANFELICE_1_IN','SANTASOFIA_1_IN','PISTOR_1_IN','APOSTOLI_1_IN','GRISOSTOMO_1_IN']].corr()
  if c<5:
    a = sns.heatmap(corrMatrix ,annot = True,cbar = False,ax = axs[c])
    a.set_title(date)
  else:
    a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax = axs[c])
    a.set_title(date)
  c = c + 1      
fig.suptitle('correlation STRADA NOVA')

plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')+ '/STRADANOVA_correlation_sim.png')
plt.show()  




fig,axs = plt.subplots(1,6,sharey = True)
c = 0

for date in start_date:
#  df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','src_Scalzi_2_IN-Scalzi_3_IN-Costituzione_1_IN-Papadopoli_1_IN---attr_Farsetti_1_IN-{}'.format(date),'output_sim_0')+'/venezia_barriers_{0}_{1}.csv'.format(date.split(' ')[0].split('-')[0][2:]+date.split(' ')[0].split('-')[1]+date.split(' ')[0].split('-')[2],date.split(' ')[1].split('-')[0]+date.split(' ')[1].split('-')[1]+date.split(' ')[1].split('-')[2]),';')
  df_day = pd.read_csv(pt_+'/venezia_barriers_210717_230000.csv',';')

  corrMatrix = df_day[['COSTITUZIONE_1_IN','PAPADOPOLI_1_IN','TREPONTI_1_IN','RAGUSEI_1_IN','RAGUSEI_2_IN','RAGUSEI_3_IN','RAGUSEI_4_IN','BARNABA_1_IN','CASINNOBILI_1_IN']].corr()
  if c<5:
    a = sns.heatmap(corrMatrix ,annot = True,cbar = False,ax = axs[c])
    a.set_title(date)
  else:
    a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax = axs[c])
    a.set_title(date)
  c = c + 1      
fig.suptitle('correlation STRADA OCCIDENTALE')
plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')+ '/STRADAPAPADOPOLI_correlation_sim.png')
plt.show()

fig,axs = plt.subplots(1,6,sharey = True)
c = 0
for date in start_date:
#  df_day = pd.read_csv(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','src_Scalzi_2_IN-Scalzi_3_IN-Costituzione_1_IN-Papadopoli_1_IN---attr_Farsetti_1_IN-{}'.format(date),'output_sim_0')+'/venezia_barriers_{0}_{1}.csv'.format(date.split(' ')[0].split('-')[0][2:]+date.split(' ')[0].split('-')[1]+date.split(' ')[0].split('-')[2],date.split(' ')[1].split('-')[0]+date.split(' ')[1].split('-')[1]+date.split(' ')[1].split('-')[2]),';')
  df_day = pd.read_csv(pt_+'/venezia_barriers_210717_230000.csv',';')
  corrMatrix = df_day[['SCALZI_2_IN','SCALZI_3_IN','SANGIACOMO_1_IN','SANAGOSTIN_1_IN','TERAANTONIO_1_IN','MADONETA_1_IN','MANDOLA_2_IN']].corr()
  if c<5:
    a = sns.heatmap(corrMatrix ,annot = True,cbar = False,ax = axs[c])
    a.set_title(date)
  else:
    a = sns.heatmap(corrMatrix ,annot = True,cbar = True,ax = axs[c])
    a.set_title(date)
  c = c + 1      
fig.suptitle('correlation VIA CENTRALE verso S.M.')
plt.savefig(os.path.join(os.environ['WORKSPACE'],'slides','work_slides','temporary_plot')+ '/VIACENTRALE_correlation_sim.png')
plt.show()