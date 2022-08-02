import argparse 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description='Insert the simulation of interest to compare with the real data')
parser.add_argument('--date',help='insert date of simulation in format %d-%m-%Y, default:2021-07-15 00:00:00 ',type=str,default='2021-07-15')
parser.add_argument('--citytag',help='insert citytag',type=str,default='venezia')
parser.add_argument('--avg_opt',help='insert average',type=bool,default=False)

args= parser.parse_args()

#INPUT
file_distances_real_data = r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\output\real_data\averaged\df_barriers_dist_week_avg.csv'
file_distances_simulation = r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\output\daily\df_barriers_dist.csv'
df_real_data = pd.read_csv(file_distances_real_data,',')
df_simulation = pd.read_csv(file_distances_simulation,',')

#os.path.join(saving_dir,'df_barriers_dist_{}.csv'.format(day.strftime('%d-%m-%Y')))
list_strada_nova=['SCALZI_2_OUT','FARSETTI_1_OUT','MADDALENA_1_OUT','SANFELICE_1_OUT','SANTASOFIA_1_OUT','PISTOR_1_OUT','APOSTOLI_1_OUT','GRISOSTOMO_2_OUT','SALVADOR_4_OUT']
list_strada_principale = ['SCALZI_2_OUT','SOTBETTINA_1_OUT','SOTCAPELER_1_OUT','RIALTO_1_OUT','SALVADOR_4_OUT']
list_terza_opzione = ['SCALZI_2_OUT','PAPADOPOLI_1_IN','MARGHERITA_1_OUT','CAFOSCARI_2_IN','MANDOLA_4_IN']
lista_quarta_opzione = ['SCALZI_2_OUT','CALLELACA_1_OUT','SANROCCO_1_OUT','MANDOLA_4_IN']
list_sum_sim = []
list_sum_real = []
list_couple_barriers =[]
for bar1 in range(len(list_strada_nova)):
    for bar2 in range(len(list_strada_nova)):
        if bar1!=bar2:
            try:
                b1=df_simulation.loc[df_simulation['barrier1']==list_strada_nova[bar1]]
                print(list_strada_nova[bar2],b1['barrier2'])
                b1 = b1.loc[b1['barrier2']==list_strada_nova[bar2]].iloc[0]['sum']
                a1=df_real_data.loc[df_real_data['barrier1']==list_strada_nova[bar1]]
                a1 = a1.loc[a1['barrier2']==list_strada_nova[bar2]].iloc[0]['sum']
                list_sum_sim.append(b1)
                list_sum_real.append(a1)
                list_couple_barriers.append(list_strada_nova[bar1].split('_')[0]+'-'+list_strada_nova[bar2].split('_')[0])
            except:
                pass
#1
list_x=np.arange(len(list_couple_barriers))
plt.plot(list_x,list_sum_real)
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3,rotation = 90)  
#2
plt.plot(list_x,list_sum_sim)
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3,rotation = 90) 
plt.title('simulated vs real data no locals barriers strada nova')
plt.legend(['real','simulation'])
plt.show()
plt.savefig(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\plots\strada_nova.png')
#3
plt.plot(list_x,np.array(list_sum_sim)/np.array(list_sum_real))
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3,rotation = 90) 
plt.title('simulated over real data no locals barriers strada nova')
plt.show()
plt.savefig(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\plots\strada_nova_rapporto.png')


list_sum_sim = []
list_sum_real = []
list_couple_barriers =[]
for bar1 in range(len(list_strada_principale)):
    for bar2 in range(len(list_strada_principale)):
        if bar1!=bar2:
            try:
                b1=df_simulation.loc[df_simulation['barrier1']==list_strada_principale[bar1]]
                b1 = b1.loc[b1['barrier2']==list_strada_principale[bar2]].iloc[0]['sum']
                a1=df_real_data.loc[df_real_data['barrier1']==list_strada_principale[bar1]]
                a1 = a1.loc[a1['barrier2']==list_strada_principale[bar2]].iloc[0]['sum']
                list_sum_sim.append(b1)
                list_sum_real.append(a1)
                list_couple_barriers.append(list_strada_principale[bar1].split('_')[0]+'-'+list_strada_principale[bar2].split('_')[0])
            except:
                pass
#1
list_x=np.arange(len(list_couple_barriers))
plt.plot(list_x,list_sum_real)
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3,rotation = 90) 
#2
plt.plot(list_x,list_sum_sim)
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3,rotation = 90) 
plt.title('simulated vs real data no locals barriers strada principale')
plt.legend(['real','simulation'])
plt.show()
plt.savefig(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\plots\strada_principale.jpg')
#2
plt.plot(list_x,np.array(list_sum_sim)/np.array(list_sum_real))
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3,rotation = 90) 
plt.title('simulated over real data no locals barriers strada principale')
plt.show()
plt.savefig(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\plots\strada_principale_rapporto.jpg')


list_sum_sim = []
list_sum_real = []
list_couple_barriers =[]
for bar1 in range(len(list_terza_opzione)):
    for bar2 in range(len(list_terza_opzione)):
        if bar1!=bar2:
            try:
                b1=df_simulation.loc[df_simulation['barrier1']==list_terza_opzione[bar1]]
                b1 = b1.loc[b1['barrier2']==list_terza_opzione[bar2]].iloc[0]['sum']
                a1=df_real_data.loc[df_real_data['barrier1']==list_terza_opzione[bar1]]
                a1 = a1.loc[a1['barrier2']==list_terza_opzione[bar2]].iloc[0]['sum']
                list_sum_sim.append(b1)
                list_sum_real.append(a1)
                list_couple_barriers.append(list_terza_opzione[bar1].split('_')[0]+'-'+list_terza_opzione[bar2].split('_')[0])
            except:
                pass
#1
list_x=np.arange(len(list_couple_barriers))
plt.plot(list_x,list_sum_real)
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3,rotation = 90) 
#2
plt.plot(list_x,list_sum_sim)
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3,rotation = 90) 
plt.title('simulated vs real  data no locals barriers terza opzione')
plt.legend(['real','simulation'])
plt.show()
plt.savefig(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\plots\terza_opzione.jpg')
#3
plt.plot(list_x,np.array(list_sum_sim)/np.array(list_sum_real))
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3,rotation = 90) 
plt.title('simulated over real data no locals barriers terza strada')
plt.show()
plt.savefig(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\plots\terza_strada_rapporto.jpg')

list_sum_sim = []
list_sum_real = []
list_couple_barriers =[]
for bar1 in range(len(lista_quarta_opzione)):
    for bar2 in range(len(lista_quarta_opzione)):
        if bar1!=bar2:
            try:
                b1=df_simulation.loc[df_simulation['barrier1']==list_quarta_opzione[bar1]]
                b1 = b1.loc[b1['barrier2']==list_quarta_opzione[bar2]].iloc[0]['sum']
                a1=df_real_data.loc[df_real_data['barrier1']==list_quarta_opzione[bar1]]
                a1 = a1.loc[a1['barrier2']==list_quarta_opzione[bar2]].iloc[0]['sum']
                list_sum_sim.append(b1)
                list_sum_real.append(a1)
                list_couple_barriers.append(list_quarta_opzione[bar1].split('_')[0]+'-'+list_quarta_opzione[bar2].split('_')[0])
            except:
                pass
#1
list_x=np.arange(len(list_couple_barriers))
plt.plot(list_x,list_sum_real)
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3, rotation = 90) 
#2
plt.plot(list_x,list_sum_sim)
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3,rotation = 90) 
plt.title('simulated vs real data no locals barriers quarta opzione')
plt.legend(['real','simulation'])
plt.show()
plt.savefig(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\plots\quarta_opzione.jpg')
#3
plt.plot(list_x,np.array(list_sum_sim)/np.array(list_sum_real))
plt.xticks(list_x, list_couple_barriers)
plt.xticks(fontsize=3,rotation = 90) 
plt.title('simulated over real data no locals barriers quarta strada')
plt.show()
plt.savefig(r'C:\Users\aamad\phd_scripts\codice\slides\work_ws\plots\quarta_strada_rapporto.jpg')
