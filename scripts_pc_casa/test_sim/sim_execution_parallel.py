import sys
import os

try:
    sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides','script_pc_casa','test_sim'))
    from setting_functions_sim import *
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e

#######################################
###### GLOBAL DIRECTORIES FOR CONF ###
#######################################
conf_dir0 = os.path.join(os.environ['WORKSPACE'],'slides','pvt','conf')
#
conf_dir=os.path.join(os.environ['WORKSPACE'],'slides','work_slides','conf_files')
dir_data = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data','barriers_config.csv')
state_basename = os.path.join(os.environ['WORKSPACE'],'slides','work_slides','output')
file_distances_real_data =os.path.join(os.environ['WORKSPACE'],'slides','work_slides','data','COVE flussi_pedonali 18-27 luglio.xlsx')
real_data=pd.read_excel(file_distances_real_data, engine='openpyxl')
time_format='%Y-%m-%d %H:%M:%S'
