import os
import sys

# from https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html

#setup
if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("please declare environment variable 'SUMO_HOME'")


#init sumo simulation
# -d, --delay FLOAT  Use FLOAT in ms as delay between simulation steps
executable = 'sumo-gui.exe' if os.name == 'nt' else 'sumo-gui'
# executable = 'sumo.exe' if os.name == 'nt' else 'sumo' # Running SUMO without the graphical interface can significantly speed up the simulation
sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', executable)
'''
Notes: 
     A smaller step length (e.g., 0.5) => more frequent updates, which can slow down the simulation but increase accuracy. Conversely, increasing it (e.g., 1.0 or more) will speed up the simulation at the cost of some detail.  '--step-length', '0.5'

'''
sumoCmd = [sumoBinary, "-c", "sumo/3_2_merge.sumocfg", '--start']