import numpy as np
import math

# Vehicle generation rates (bimodal distribution pattern)
car_generation_rates_per_lane = [
            75, 50,     # 00:00-00:30-01:00
            50, 25,     # 01:00-01:30-02:00
            25, 10,     # 02:00-02:30-03:00
            10, 15,     # 03:00-03:30-04:00 
            35, 50,     # 04:00-04:30-05:00
            75, 125,    # 05:00-05:30-06:00
            300, 480,   # 06:00-06:30-07:00
            900, 1115,  # 07:00-07:30-08:00
            1275, 1380, # 08:00-08:30-09:00
            1225, 1100, # 09:00-09:30-10:00
            1090, 1075, # 10:00-10:30-11:00
            1200, 1325, # 11:00-11:30-12:00
            1425, 1525, # 12:00-12:30-13:00
            1515, 1500, # 13:00-13:30-14:00
            1400, 1300, # 14:00-14:30-15:00
            1290, 1280, # 15:00-15:30-16:00
            1375, 1400, # 16:00-16:30-17:00
            1575, 1725, # 17:00-17:30-18:00
            1575, 1400, # 18:00-18:30-19:00
            1200, 1125, # 19:00-19:30-20:00
            850, 700,   # 20:00-20:30-21:00
            575, 450,   # 21:00-21:30-22:00
            350, 225,   # 22:00-22:30-23:00
            200, 100    # 23:00-23:30-00:00
]

# Day of the week factor # TODO: add this one in flow generation
day_of_the_week_factor = [
    np.random.triangular(0.95, 1, 1.05),   # Monday
    np.random.triangular(0.90, 1, 1.10),   # Tuesday
    np.random.triangular(0.90, 1, 1.10),   # Wednesday
    np.random.triangular(0.90, 1, 1.10),   # Thursday
    np.random.triangular(0.95, 1, 1.05), # Friday
    np.random.triangular(0.80, 1, 1.2), # Saturday
    np.random.triangular(0.80, 1, 1.2)    # Sunday
]

# Day off factor (assuming no day off effect) # TODO: add this one in flow generation
day_off_factor = [
    1.0, # Monday
    1.0, # Tuesday
    1.0, # Wednesday
    1.0, # Thursday
    1.0, # Friday
    1.0, # Saturday
    1.0  # Sunday
]

# Define route and other common flow attributes
route_id = "r_0"
depart_lane = "free"
depart_pos = "free"
depart_speed = "speedLimit"
lanes = 3

def flow_generation(base_traffic_jam_exponent, day_index = 0):
    # Vehicle type distributions
    truck = np.random.uniform(10, 15) * (1.0 if np.random.triangular(0, 0.86, 1) > 0.5 else 0.0)
    cars = np.random.uniform(70, 85) * 1.15
    motorcycle = 0
    bus = 0
    van = 0

    if (truck + cars < 92.5):
        remaining_veh = 100 - truck - cars
        van = np.random.uniform(min(9, remaining_veh), min(12, remaining_veh))
        remaining_veh = 100 - truck - cars - van
        if (remaining_veh < 95):
            bus = np.random.uniform(remaining_veh * 0.95, remaining_veh)
            motorcycle = max(0, remaining_veh - bus)
        else:
            truck = truck * 0.95
            cars = cars * 0.95
            remaining_veh = 100 - truck - cars
            van = np.random.uniform(remaining_veh * 0.75, remaining_veh * 0.95)
            remaining_veh = 100 - truck - cars - van
            bus = np.random.uniform(remaining_veh * 0.90, remaining_veh * 0.98)
            motorcycle = remaining_veh - bus
    else:
        truck = truck * 0.925
        cars = cars * 0.925
        remaining_veh = 100 - truck - cars
        van = np.random.uniform(remaining_veh * 0.75, remaining_veh * 0.95)
        remaining_veh = 100 - truck - cars - van
        bus = np.random.uniform(remaining_veh * 0.90, remaining_veh * 0.98)
        motorcycle = remaining_veh - bus
    
    normal_car = cars * np.random.uniform(0.75, 0.95)
    fast_car = cars - normal_car

    # Calculate proportions
    total_distribution = normal_car + fast_car + van + bus + motorcycle + truck
    proportions = {
        "normal_car": normal_car / total_distribution,
        "fast_car": fast_car / total_distribution,
        "van": van / total_distribution,
        "bus": bus / total_distribution,
        "motorcycle": motorcycle / total_distribution,
        "truck": truck / total_distribution
    }

    # Open a .rou.xml file to write flows
    with open("./sumo/generated_flows.rou.xml", "w") as f:
        edges = "seg_10_before seg_9_before seg_8_before seg_7_before seg_6_before seg_5_before seg_4_before seg_3_before seg_2_before seg_1_before seg_0_before seg_0_after seg_1_after"
        flows = [] # Collect flows here

        # Iterate over each pair of rates
        for i in range(0, len(car_generation_rates_per_lane), 2):
            # Calculate start and end times for each flow
            begin_time = i * 1800

            # Get vehsPerHour for current interval
            low = math.exp(base_traffic_jam_exponent) * 0.75
            high = math.exp(base_traffic_jam_exponent) * 1.25
            mid = math.exp(base_traffic_jam_exponent)
            traffic_jam_factor = np.random.triangular(low, mid, high)
            vehs_per_hour_1 = car_generation_rates_per_lane[i] * traffic_jam_factor
            vehs_per_hour_2 = car_generation_rates_per_lane[i+1] * traffic_jam_factor
            
            # Calculate the flow index based on the current iteration
            flow_index = i // 2
            
            # Create flows for each vehicle type based on their proportions
            for vehicle_type in proportions:
                vehs_1 = vehs_per_hour_1 * proportions[vehicle_type]
                vehs_2 = vehs_per_hour_2 * proportions[vehicle_type]
                
                if vehs_1 > 0:
                    flows.append((begin_time,
                                f'<flow id="{vehicle_type}_{flow_index}_0_{vehicle_type}" type="{vehicle_type}" begin="{begin_time}" end="{begin_time + 1800}" '
                                f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                f'route="{route_id}" vehsPerHour="{vehs_1}"/>\n'))
                
                if vehs_2 > 0:
                    flows.append((begin_time + (30 * len(car_generation_rates_per_lane)),
                                f'<flow id="{vehicle_type}_{flow_index}_1_{vehicle_type}" type="{vehicle_type}" begin="{begin_time + 1800}" end="{begin_time + 3600}" '
                                f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                f'route="{route_id}" vehsPerHour="{vehs_2}"/>\n'))

        # Sort flows by begin time
        flows.sort(key=lambda x: x[0])

        # Open a .rou.xml file to write sorted flows
        with open("./sumo/generated_flows.rou.xml", "w") as f:
            f.write('<routes>\n')
            f.write('\n')
            
            # Define specific vehicle types
            f.write('    <vType id="normal_car" accel="2.6" decel="4.5" sigma="0.5" length="4.5" maxSpeed="180" color="1,0,0"/>\n')
            f.write('    <vType id="fast_car" accel="3.2" decel="4.9" sigma="0.4" length="4.2" maxSpeed="240" color="0.8,0.8,0"/>\n')
            f.write('    <vType id="van" accel="1.8" decel="3.2" sigma="0.6" length="6.2" maxSpeed="130" color="1,1,1"/>\n')
            f.write('    <vType id="bus" accel="1.2" decel="3.8" sigma="0.7" length="12.0" maxSpeed="100" color="1,.5,.5"/>\n')
            f.write('    <vType id="motorcycle" accel="3.8" decel="6.2" sigma="0.3" length="2.2" maxSpeed="180" color=".2,.2,.8"/>\n')
            f.write('    <vType id="truck" accel="1.1" decel="3.1" sigma="0.9" length="10.5" maxSpeed="100" color=".4,.4,.4"/>\n')

            f.write('\n')
            f.write(f'    <route id="{route_id}" edges="{edges}"/>\n') # Replace {your_edges_here} with actual edges
            f.write('\n')

            # Write sorted flows to file
            for _, flow in flows:
                f.write(flow)

            f.write('</routes>\n')

        print("Flow generation complete.")