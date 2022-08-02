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

try:
    sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides','script_pc_casa','test_sim'))
    import simulator
    import sim_objects
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e



try:
  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'bin'))
  from pysim import simulation

  sys.path.append(os.path.join(os.environ['WORKSPACE'], 'slides', 'python'))
  from conf import conf
except Exception as e:
  raise Exception('library loading error : {}'.format(e)) from e


if __name__ == '__main__':
    # initialize simulation parameters
    sim = simulator()
    sim.pick_day_input()
    sim.averaging_fluxes()
    sim.set_simcfg_fluxes_from_list_sources()
    # copying the objects from metadata
    ch = configuration_handler()
    ch.assign_sources_json(sim.simcfgorig)
    ch.assign_new_sources(sim.list_added_sources, sim.data_barriers,sim.df_avg)
    ch.reset_sources(sim.reset_source_list,sim.data_barriers)
    ch.assign_attractions()
    ch.is_locals(sim.attraction_activate,
                sim.locals_,
                sim.local_distribution,
                sim.name_attracctions,
                sim.add_attractions_bool)
    ch.assign_new_sources(sim.new_source_list,
                   sim.new_source_bool,
                   sim.reset_source_list,
                   sim.reset_source_bool,
                   sim.change_source_list,
                   sim.change_source_bool)
    # run the simulation
    
    
    # clustering procedure
    
    
    # plotting procedure
    