import os, sys
from statistics import mean
import traci
import numpy as np
from matplotlib import pyplot as plt
import control_mechanisms
import pandas as pd

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
sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', executable)
sumoCmd = [sumoBinary, "-c", "2_1_merge.sumocfg", '--start']

#run sumo simulation
traci.start(sumoCmd)

# ----------------------------------------------- VARIABLE SETTING -----------------------------------------------

edges = ["seg_10_before","seg_9_before","seg_8_before","seg_7_before","seg_6_before","seg_5_before","seg_4_before","seg_3_before","seg_2_before","seg_1_before","seg_0_before","seg_0_after","seg_1_after"]

# LANES & MAXIMUM SPEEDS
# naming:
# laneSEGment_"number, 0 means closest to merge zone"_"before or after the merge"_"lane number counting from bottom to top"
seg_10_before = ["seg_10_before_2", "seg_10_before_1", "seg_10_before_0"]
seg_9_before = ["seg_9_before_2", "seg_9_before_1", "seg_9_before_0"]
seg_8_before = ["seg_8_before_2", "seg_8_before_1", "seg_8_before_0"]
seg_7_before = ["seg_7_before_2", "seg_7_before_1", "seg_7_before_0"]
seg_6_before = ["seg_0_before_2", "seg_6_before_1", "seg_6_before_0"]
seg_5_before = ["seg_5_before_2", "seg_5_before_1", "seg_5_before_0"]
seg_4_before = ["seg_4_before_2", "seg_4_before_1", "seg_4_before_0"]
seg_3_before = ["seg_3_before_2", "seg_3_before_1", "seg_3_before_0"]
seg_2_before = ["seg_2_before_2", "seg_2_before_1", "seg_2_before_0"]
seg_1_before = ["seg_1_before_2", "seg_1_before_1", "seg_1_before_0"]
seg_0_before = ["seg_0_before_2", "seg_0_before_1", "seg_0_before_0"]
segments_before = [seg_10_before, seg_9_before, seg_8_before, seg_7_before, seg_6_before, seg_5_before, seg_4_before, seg_3_before, seg_2_before, seg_1_before, seg_0_before]

seg_0_after = ["seg_0_after_1", "seg_0_after_0"]
seg_1_after = ["seg_1_after_1", "seg_1_after_0"]
segments_after = [seg_0_after, seg_1_after]
all_segments = segments_before + segments_after
print(all_segments)

low_speed = 15 # 50 km/h
speed_max = 33.33 # 120 km/h

[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_10_before]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_9_before]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_8_before]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_7_before]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_6_before]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_5_before]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_4_before]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_3_before]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_2_before]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_1_before]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_0_before]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_0_after]
[traci.lane.setMaxSpeed(lane, speed_max) for lane in seg_1_after]


# ROAD SENSORS / INDUCTION LOOPS
# defined in additional.xml
# naming:
# equal to lanes
# keeping to the sumo objects used
# loop = induction loop ... for measurements at a point
# detector = lane are detector ... for measurements along a lane

loops_beforeA = ["loop_seg_0_before_2A", "loop_seg_0_before_1A", "loop_seg_0_before_0A"]
loops_beforeB = ["loop_seg_0_before_2B", "loop_seg_0_before_1B", "loop_seg_0_before_0B"]
loops_beforeC = ["loop_seg_0_before_2C", "loop_seg_0_before_1C", "loop_seg_0_before_0C"]
loops_beforeD = ["loop_seg_0_before_2D", "loop_seg_0_before_1D", "loop_seg_0_before_0D"]
loops_before = [loops_beforeA, loops_beforeB, loops_beforeC, loops_beforeD]
detectors_before = ["detector_seg_0_before_2", "detector_seg_0_before_1", "detector_seg_0_before_0"]

loops_after = ["loop_seg_0_after_1", "loop_seg_0_after_0"]
detectors_after = ["detector_seg_0_after_1", "detector_seg_0_after_0"]

detector_length = 50 #meters

# INTRODUCE METRICS
density = 0
flow = 0
mean_speed = 0

# overall road speed stuff
mean_edge_speed = np.zeros(len(edges)) # for each edge
mean_road_speed = 0
ms = [] # mean speeds

cvs_seg_time = []
for i in range(len(all_segments)):
    cvs_seg_time.append([])
print(cvs_seg_time)

emissions = np.zeros(len(edges)) # for each edge
emissions_over_time = [] # for each time step


occupancy = 0 # highest anywhere in segment before merge
num = 0

density_before = 0
# ~ think about: travel time as overall satisfaction metric

# metric accumulators
veh_time_sum = 0 # for vehicles passing a point
veh_space_sum = 0 # for vehicles passing an area
mean_speed_sum = 0

veh_space_before_sum = 0
occupancy_sum = 0
num_sum = 0
dens = []
occ = []
flw = []

# CONTROL MECHANISM PARAMETERS
#mtfc
b = 1.0
application_area = segments_before[9:10]
print(application_area)

# SIMULATION PARAMETERS
step = 0
aggregation_time = 30 # seconds - always aggregate the last 100 step to make decision in the present
previous_harm_speeds = [120] * int(len(segments_before) / 2)

# ----------------------------------------------- SIMULATION LOOP -----------------------------------------------

# run till all cars are gone
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep() 
    step += 1
    
    # GATHER METRICS FROM SENSORS    
    # for some it is important to average over the number of lanes on the edge

    # AFTER
    veh_space_sum += sum([traci.lanearea.getLastStepVehicleNumber(detector) for detector in detectors_after])
    veh_time_sum += sum([traci.inductionloop.getLastStepVehicleNumber(loop) for loop in loops_after])

    mean_speed = sum([traci.inductionloop.getLastStepMeanSpeed(loop) for loop in loops_after]) / len(loops_after)
    # speed -1 indicated no vehicle on the loop
    if mean_speed >= 0:
        mean_speed_sum += mean_speed

    emission_sum = 0
    for i, edge in enumerate(edges):   
        mean_edge_speed[i] += traci.edge.getLastStepMeanSpeed(edge)
        emission_sum += traci.edge.getCO2Emission(edge)
    
    emissions_over_time.append(emission_sum)
    

    # BEFORE
    veh_space_before_sum += sum([traci.lanearea.getLastStepVehicleNumber(detector) for detector in detectors_before])

    # collecting the number of vehicles and occupancy right in front (or even in) of the merge area
    # choose max occupancy of a few sensors
    occ_max = 0
    for loops in loops_before:
        occ_loop = sum([traci.inductionloop.getLastStepOccupancy(loop) for loop in loops]) / len(loops)
        if occ_loop > occ_max:
            occ_max = occ_loop

    occupancy_sum += occ_max
    num_sum += sum([traci.inductionloop.getLastStepVehicleNumber(loop) for loop in loops_before[0]]) # only at one sensor
    
    # EVALUATE THE METRICS FREQUENTLY AND USE TO ADJUST VARIABLE SPEED LIMITS
    if step % aggregation_time == 0:

        # collected metrics are devided by the aggregation time to get the average values
        # OVERALL
        mean_edge_speed = mean_edge_speed / aggregation_time # first is acutally a sum
        mean_road_speed = sum(mean_edge_speed) / len(mean_edge_speed)
        ms.append(mean_road_speed)

        # AFTER THE MERGE
        density = ((veh_space_sum / aggregation_time) / detector_length) * 1000
        flow = (veh_time_sum / aggregation_time) * 3600
        flw.append(flow)
        mean_speed = (mean_speed_sum / aggregation_time) * 3.6 # one speed metric is enough - equals to spot speed

        # BEFORE THE MERGE
        density_before = ((veh_space_before_sum / aggregation_time) / detector_length) * 1000
        dens.append(density_before)

        occupancy = occupancy_sum / aggregation_time
        occ.append(occupancy)
        num = num_sum / aggregation_time

        # monitor the safety of road segments (CVS) - stores cvs value for each segment for each aggregation time step
        for i, seg in enumerate(all_segments):
            print(i, ':', seg)
            cvs_sum = 0
            for lane in seg:
                # for cvs
                ids = traci.lane.getLastStepVehicleIDs(lane)
                speeds = []
                for id in ids:
                    speeds.append(traci.vehicle.getSpeed(id))
                speeds = np.array(speeds)
                lane_avg = np.mean(speeds)
                lane_stdv = np.std(speeds)
                cvs_sum += lane_stdv / lane_avg
            cvs_seg = cvs_sum / len(seg)
            if np.isnan(cvs_seg):
                cvs_seg = 0
            cvs_seg_time[i].append(cvs_seg)
        
        # CONTROL MECHANISM - VARIABLE SPEED LIMIT ALGORITHM
        # implementation of https://www.sciencedirect.com/science/article/pii/S0968090X07000873        
        
        #control_mechanisms.lecture_mechanism(occupancy_desired=11, occupancy_old=occupancy, flow_old=flow, road_segments=segments_before[:10])  

        #b = control_mechanisms.mtfc(occupancy, 12, b, speed_max, application_area)
        
        #previous_harm_speeds = control_mechanisms.adjusted_mcs(segments_before, speed_max, previous_harm_speeds)

        # reset accumulator
        veh_time_sum = 0
        veh_space_sum = 0
        mean_speed_sum = 0

        mean_edge_speed = np.zeros(len(edges))
        mean_road_speed = 0

        veh_space_before_sum = 0
        occupancy_sum = 0
        num_sum = 0

# Save metrics into csv file.
approach = 'baseline'
with open(f'./metrics/{approach}_metrics.csv', 'w+') as metrics_file:
    list_to_string = lambda x: ','.join([ str(elem) for elem in x ]) + '\n'
    metrics_file.write(list_to_string(ms))
    metrics_file.write(list_to_string(flw))
    metrics_file.write(list_to_string(emissions_over_time))

# plot occupancy and flow diagram to get capacity flow    
fig, ax = plt.subplots(1,1, figsize=(15,30)) 
#plt.xticks(np.arange(min(dens), max(dens)+1, 1.0))
#plt.plot(dens, flw, 'bo', )
#plt.show()

print("CO2",emissions)
print('FLOW MAX',max(flw))
print(type(cvs_seg_time[0][0]))
print(cvs_seg_time[0][0])
print(cvs_seg_time)
pd.DataFrame(cvs_seg_time).to_csv(f'./metrics/{approach}_cvs.csv', index=False, header=False)


plt.xticks(np.arange(min(occ), max(occ)+1, 1.0))
plt.plot(occ, flw, 'bo')
plt.show() 
plt.plot(ms)
plt.show()
plt.plot(emissions_over_time)
plt.show()
plt.plot(flw)
plt.show()

traci.close()