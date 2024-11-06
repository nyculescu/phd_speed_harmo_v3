import numpy as np
import math
import logging
from config import *
from scipy.ndimage import gaussian_filter

# Adjust the amplitude of the daily pattern
def adjust_amplitude(dly_pattern, amplitude_factor):
    return [round(value * math.exp(amplitude_factor * (value / max(dly_pattern)))) for value in dly_pattern]

# Define route and other common flow attributes
route_id = "r_0"
depart_lane = "free"
depart_pos = "free"
depart_speed = "speedLimit"
lanes = 3

def bimodal_distribution(x = np.arange(0, 24.5, 0.5)): # 24.5 is used to include 24:00
    baseline = np.random.uniform(0.001, 0.025) 
    mu1 = 8.0 + np.random.uniform(-0.5, 0.5) # Morning peak
    mu2 = 17.0 + np.random.uniform(-0.5, 0.5)# Evening peak
    sigma1 = 2.0 + np.random.uniform(-0.2, 0.2) # Range for morning peak width
    sigma2 = 2.0 + np.random.uniform(-0.2, 0.2) # Range for evening peak width
    A1 = 1.0 + np.random.uniform(-0.05, 0.05) # Amplitude for morning peak
    A2 = 0.8 + np.random.uniform(-0.04, 0.04) # Amplitude for evening peak
    # Compute the bimodal distribution
    y = (
        A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) +
        A2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)) + 
        baseline
    )
    if np.random.rand() < 0.5:  # 25% chance to add a third peak
        mu3 = 13.0 + np.random.uniform(-1.0, 1.0)  # Midday peak
        sigma3 = np.random.uniform(0.5, 1.0)
        A3 = np.random.uniform(0.3, 0.6)
        y += A3 * np.exp(-((x - mu3) ** 2) / (2 * sigma3 ** 2))
    y = y * (1 + gaussian_filter(np.random.normal(0, 0.02, size=y.shape), sigma=2)) # Add smoothed random noise to simulate natural fluctuations
    y = y / np.max(y) # Normalize to 1
    y = y * base_demand # Scale up to base demand
    return y.astype(int)[:48] # Return first 48 entries for 24 hours at half-hour intervals

def triangular_distribution(x=np.arange(0, 24.5, 0.5)):
    # Define the peak time for traffic
    peak_time = 13.0 + np.random.uniform(-1.0, 1.0)  # Slight randomness in peak time
    morning_start = 6.0 + np.random.uniform(-0.5, 0.5)  # Start of morning traffic
    evening_end = 19.0 + np.random.uniform(-0.5, 0.5)   # End of evening traffic
    # Create a triangular distribution
    y = np.piecewise(
        x,
        [x <= peak_time, x > peak_time],
        [
            lambda x: np.clip((x - morning_start) / (peak_time - morning_start), 0, None),  # Rising slope
            lambda x: np.clip((evening_end - x) / (evening_end - peak_time), 0, None)       # Falling slope
        ]
    )
    # Add random noise and smooth it to simulate natural fluctuations
    y = y * (1 + gaussian_filter(np.random.normal(0, 0.02, size=y.shape), sigma=2))
    # Normalize to base demand
    y = y / np.max(y) * base_demand
    # Ensure no negative values and set minimum values between (0, 50)
    y[y < 50] = np.random.uniform(0, 50)
    return y.astype(int)[:48]  # Return first 48 entries for half-hour intervals

def flow_generation_wrapper(daily_pattern_amplitude, model, idx, num_days):
    if mock_days_and_weeks:
        dly_pattern = mock_daily_pattern
    else:
        dly_pattern = triangular_distribution() if np.random.choice([False, True]) else bimodal_distribution()
    daily_pattern_ampl = adjust_amplitude(dly_pattern, daily_pattern_amplitude)

    if model == "all":
        # Generate content for DQN model
        flow_generation(all_models[0], 0, daily_pattern_ampl, num_days)
        
        # Read the content from the DQN file
        with open(f"./traffic_environment/sumo/generated_flows_{all_models[0]}_{0}.rou.xml", 'r') as f:
            firstmodel_file_path_content = f.read()

        # Write the content to the files for the other models (except the first one)
        for m in all_models[1:]:
            file_path = f"./traffic_environment/sumo/generated_flows_{m}_{0}.rou.xml"
            with open(file_path, 'w') as f:
                f.write(firstmodel_file_path_content)
    elif model in all_models:
        flow_generation(model, idx, daily_pattern_ampl, num_days)
    else:
        logging.error(f"Model {model} is not supported. Supported models are: {all_models}")

def flow_generation(model, idx, daily_pattern_ampl, num_days):
    # Open a .rou.xml file to write flows
    with open(f"./traffic_environment/sumo/generated_flows_{model}_{idx}.rou.xml", "w") as f:
        edges = "seg_10_before seg_9_before seg_8_before seg_7_before seg_6_before seg_5_before seg_4_before seg_3_before seg_2_before seg_1_before seg_0_before seg_0_after seg_1_after"
        flows = [] # Collect flows here

        # Iterate over each pair of rates
        for day_index in range(0, num_days):
            for i in range(0, len(daily_pattern_ampl), 2):
                # Vehicle type distributions
                trucks = np.random.uniform(10, 15) * (1.0 if (day_index == 6) else 0.0)
                cars = np.random.uniform(70, 85) * 1.15
                motorcycle = 0
                bus = 0
                van = 0

                if (trucks + cars < 92.5):
                    remaining_veh = 100 - trucks - cars
                    van = np.random.uniform(min(9, remaining_veh), min(12, remaining_veh))
                    remaining_veh = 100 - trucks - cars - van
                    if (remaining_veh < 95):
                        bus = np.random.uniform(remaining_veh * 0.95, remaining_veh)
                        motorcycle = max(0, remaining_veh - bus)
                    else:
                        trucks = trucks * 0.95
                        cars = cars * 0.95
                        remaining_veh = 100 - trucks - cars
                        van = np.random.uniform(remaining_veh * 0.75, remaining_veh * 0.95)
                        remaining_veh = 100 - trucks - cars - van
                        bus = np.random.uniform(remaining_veh * 0.90, remaining_veh * 0.98)
                        motorcycle = remaining_veh - bus
                else:
                    trucks = trucks * 0.925
                    cars = cars * 0.925
                    remaining_veh = 100 - trucks - cars
                    van = np.random.uniform(remaining_veh * 0.75, remaining_veh * 0.95)
                    remaining_veh = 100 - trucks - cars - van
                    bus = np.random.uniform(remaining_veh * 0.90, remaining_veh * 0.98)
                    motorcycle = remaining_veh - bus
                
                normal_car = cars * np.random.uniform(0.75, 0.95)
                fast_car = cars - normal_car
                trailer = trucks * 0.98
                truck = trucks - trailer

                # Calculate proportions
                total_distribution = normal_car + fast_car + van + bus + motorcycle + trucks

                if addDisobedientVehicles:
                    disobedient_normal_car_proportion = np.random.uniform(0.01, 0.1)
                    disobedient_fast_car_proportion = np.random.uniform(0.5, 0.2)
                    disobedient_van_proportion = np.random.uniform(0.01, 0.5)
                    disobedient_bus_proportion = np.random.uniform(0, 0.01)
                    disobedient_motorcycle_proportion = np.random.uniform(0, 0.06)
                    disobedient_truck_proportion = np.random.uniform(0, 0.02)
                    disobedient_trailer_proportion = np.random.uniform(0, 0.005)
                else:
                    disobedient_normal_car_proportion = 0
                    disobedient_fast_car_proportion = 0
                    disobedient_van_proportion = 0
                    disobedient_bus_proportion = 0
                    disobedient_motorcycle_proportion = 0
                    disobedient_truck_proportion = 0
                    disobedient_trailer_proportion = 0

                if addElectricVehicles:
                    electric_normal_car_proportion = np.random.uniform(0.05, 0.1)
                    electric_fast_car_proportion = np.random.uniform(0.07, 0.1)
                    electric_van_proportion = np.random.uniform(0.02, 0.04)
                    electric_bus_proportion = np.random.uniform(0.005, 0.01)
                    electric_motorcycle_proportion = np.random.uniform(0.03, 0.06)
                    electric_truck_proportion = np.random.uniform(0.01, 0.02)
                    electric_trailer_proportion = np.random.uniform(0.0025, 0.005)
                else:
                    electric_normal_car_proportion = 0
                    electric_fast_car_proportion = 0
                    electric_van_proportion = 0
                    electric_bus_proportion = 0
                    electric_motorcycle_proportion = 0
                    electric_truck_proportion = 0
                    electric_trailer_proportion = 0

                proportions = {
                    "passenger": (normal_car-disobedient_normal_car_proportion-electric_normal_car_proportion) / total_distribution,
                    "passenger/hatchback": (fast_car-disobedient_fast_car_proportion-electric_fast_car_proportion) / total_distribution,
                    "passenger/van": (van-disobedient_van_proportion-electric_van_proportion) / total_distribution,
                    "bus": (bus-disobedient_bus_proportion-electric_bus_proportion) / total_distribution,
                    "motorcycle": (motorcycle-disobedient_motorcycle_proportion-electric_motorcycle_proportion) / total_distribution,
                    "truck": (truck-disobedient_truck_proportion-electric_truck_proportion) / total_distribution,
                    "truck/trailer": (trailer-disobedient_trailer_proportion-electric_trailer_proportion) / total_distribution,

                    # Disobedient proportions:
                    "disobedient_passenger": disobedient_normal_car_proportion / total_distribution,
                    "disobedient_passenger/hatchback": disobedient_fast_car_proportion / total_distribution,
                    "disobedient_passenger/van": disobedient_van_proportion / total_distribution,
                    "disobedient_bus": disobedient_bus_proportion / total_distribution,
                    "disobedient_motorcycle": disobedient_motorcycle_proportion / total_distribution,
                    "disobedient_truck": disobedient_truck_proportion / total_distribution,
                    "disobedient_truck/trailer": disobedient_trailer_proportion / total_distribution,

                    # Electric proportions:
                    "electric_passenger": electric_normal_car_proportion / total_distribution,
                    "electric_passenger/hatchback": electric_fast_car_proportion / total_distribution,
                    "electric_passenger/van": electric_van_proportion / total_distribution,
                    "electric_bus": electric_bus_proportion / total_distribution,
                    "electric_motorcycle": electric_motorcycle_proportion / total_distribution,
                    "electric_truck": electric_truck_proportion / total_distribution,
                    "electric_truck/trailer": electric_trailer_proportion / total_distribution,
                }

                # Calculate start and end times for each flow
                begin_time = (day_index * len(daily_pattern_ampl) * 1800) + (i * 1800)

                # Get vehsPerHour for current interval
                vehs_per_hour_1 = daily_pattern_ampl[i] * day_of_the_week_factor[day_index]
                vehs_per_hour_2 = daily_pattern_ampl[i+1] * day_of_the_week_factor[day_index]
                
                # Calculate the flow index based on the current iteration
                flow_index = i // 2
                
                # Create flows for each vehicle type based on their proportions
                for vehicle_type in proportions:
                    vehs_1 = vehs_per_hour_1 * proportions[vehicle_type]
                    vehs_2 = vehs_per_hour_2 * proportions[vehicle_type]
                    
                    if vehs_1 > 0:
                        if "disobedient" in vehicle_type and addDisobedientVehicles:
                            flows.append((day_index, begin_time,
                                        f'    <flow id="{vehicle_type}_flow_{flow_index}_day_{day_index}_halfhr_0" type="{vehicle_type}" begin="{begin_time}" end="{begin_time + 1800}" '
                                        f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                        f'route="{route_id}" vehsPerHour="{vehs_1}" guiShape="{vehicle_type.removeprefix("disobedient_")}"/>\n'))
                        elif "electric" in vehicle_type and addElectricVehicles:
                            flows.append((day_index, begin_time,
                                        f'    <flow id="{vehicle_type}_flow_{flow_index}_day_{day_index}_halfhr_0" type="{vehicle_type}" begin="{begin_time}" end="{begin_time + 1800}" '
                                        f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                        f'route="{route_id}" vehsPerHour="{vehs_1}" guiShape="{vehicle_type.removeprefix("electric_")}"/>\n'))
                        else:
                            flows.append((day_index, begin_time,
                                        f'    <flow id="{vehicle_type}_flow_{flow_index}_day_{day_index}_halfhr_0" type="{vehicle_type}" begin="{begin_time}" end="{begin_time + 1800}" '
                                        f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                        f'route="{route_id}" vehsPerHour="{vehs_1}" guiShape="{vehicle_type}"/>\n'))
                    
                    if vehs_2 > 0:
                        if "disobedient" in vehicle_type and addDisobedientVehicles:
                            flows.append((day_index, begin_time + (30 * len(daily_pattern_ampl)),
                                    f'    <flow id="{vehicle_type}_flow_{flow_index}_day_{day_index}_halfhr_1" type="{vehicle_type}" begin="{begin_time + 1800}" end="{begin_time + 3600}" '
                                    f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                    f'route="{route_id}" vehsPerHour="{vehs_2}" guiShape="{vehicle_type.removeprefix("disobedient_")}"/>\n'))
                        elif "electric" in vehicle_type and addElectricVehicles:
                            flows.append((day_index, begin_time + (30 * len(daily_pattern_ampl)),
                                    f'    <flow id="{vehicle_type}_flow_{flow_index}_day_{day_index}_halfhr_1" type="{vehicle_type}" begin="{begin_time + 1800}" end="{begin_time + 3600}" '
                                    f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                    f'route="{route_id}" vehsPerHour="{vehs_1}" guiShape="{vehicle_type.removeprefix("electric_")}"/>\n'))
                        else:
                            flows.append((day_index, begin_time + (30 * len(daily_pattern_ampl)),
                                    f'    <flow id="{vehicle_type}_flow_{flow_index}_day_{day_index}_halfhr_1" type="{vehicle_type}" begin="{begin_time + 1800}" end="{begin_time + 3600}" '
                                    f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                    f'route="{route_id}" vehsPerHour="{vehs_2}" guiShape="{vehicle_type}"/>\n'))

        # Sort flows by begin time only (no need to sort by day index first)
        flows.sort(key=lambda x: (x[1]))

        # Write sorted flows
        f.write('<routes>\n')
        f.write('\n')
        
        """ Define vehicle types """
        car_following_model = 'Krauss' # https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#car-following_models
        # Define specific vehicle types
        accel = np.random.uniform(2.4, 2.8)
        decel = np.random.uniform(4.3, 4.7)
        length = np.random.triangular(3.7, 4.75, 5.7) * np.random.uniform(0.97, 1.07)
        emissionClass = "HBEFA4/PC_petrol_Euro-4"
        f.write(f'    <vType id="passenger" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.5)}" length="{length}" maxSpeed="180" color="1,0,0" personNumber="{(int)(np.random.uniform(1, 5))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.0, 3.4)
        decel = np.random.uniform(4.5, 5.0)
        length = np.random.triangular(3.7, 4.4, 5.0) * np.random.uniform(0.98, 1.02)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="passenger/hatchback" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.7)}" length="{length}" maxSpeed="240" color="0.8,0.8,0" personNumber="{(int)(np.random.uniform(1, 3))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.5, 2.0)
        decel = np.random.uniform(3.0, 3.4)
        length = np.random.triangular(3.7, 4.5, 5.0) * np.random.uniform(0.98, 1.02)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="passenger/van" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.5)}" length="{length}" maxSpeed="130" color="1,1,1" personNumber="{(int)(np.random.uniform(1, 12))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.5)
        decel = np.random.uniform(3.5, 4.1)
        length = np.random.triangular(12.0, 12.5, 18.0) * np.random.uniform(0.98, 1.02)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="bus" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.3)}" length="{length}" maxSpeed="100" color="1,.5,.5" personNumber="{(int)(np.random.uniform(1, 40))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.5, 4.1)
        decel = np.random.uniform(5.0, 6.4)
        length = np.random.uniform(2.2, 2.6)
        emissionClass = "HBEFA4/PC_petrol_Euro-6ab" # UBus_Electric_Artic_gt18t
        f.write(f'    <vType id="motorcycle" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.35)}" length="{length}" maxSpeed="180" color=".2,.2,.8" personNumber="{(int)(np.random.uniform(1, 2))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.4, 1.8)
        decel = np.random.uniform(3.3, 3.8)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="truck" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.35)}" length="8" maxSpeed="100" color=".4,.1,.4" personNumber="{(int)(np.random.uniform(1, 3))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.4)
        decel = np.random.uniform(3.0, 3.4)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="truck/trailer" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.3)}" length="13.6" maxSpeed="100" color=".4,.4,.4" personNumber="{(int)(np.random.uniform(1, 2))}" emissionClass="{emissionClass}"/>\n')

        """ Disobedient vehicle types """
        accel = np.random.uniform(2.4, 2.8)
        decel = np.random.uniform(4.3, 4.7)
        length = np.random.triangular(3.7, 4.75, 5.7) * np.random.uniform(0.97, 1.07)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        emissionClass = "HBEFA4/PC_petrol_Euro-4"
        f.write(f'    <vType id="disobedient_passenger" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="200" color="1,0,0" personNumber="{(int)(np.random.uniform(1, 5))}" sigma="{sigma}" impatience="{impatience}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.0, 3.4)
        decel = np.random.uniform(4.5, 5.0)
        length = np.random.triangular(3.7, 4.4, 5.0) * np.random.uniform(0.98, 1.02)
        impatience = np.random.uniform(0.75, 1)
        sigma=np.random.uniform(0.85, 1)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="disobedient_passenger/hatchback" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="240" color="0.8,0.8,0" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{sigma}" impatience="{impatience}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.5, 2.0)
        decel = np.random.uniform(3.0, 3.4)
        length = np.random.triangular(3.7, 4.5, 5.0) * np.random.uniform(0.98, 1.02)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="disobedient_passenger/van" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="130" color="1,1,1" personNumber="{(int)(np.random.uniform(1, 12))}" sigma="{sigma}" impatience="{impatience}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.5)
        decel = np.random.uniform(3.5, 4.1)
        length = np.random.triangular(12.0, 12.5, 18.0) * np.random.uniform(0.98, 1.02)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="disobedient_bus" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="100" color="1,.5,.5" personNumber="{(int)(np.random.uniform(1, 40))}" sigma="{sigma}" impatience="{impatience}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.5, 4.1)
        decel = np.random.uniform(5.0, 6.4)
        length = np.random.uniform(2.2, 2.6)
        impatience = np.random.uniform(0.75, 1)
        sigma=np.random.uniform(0.75, 1)
        emissionClass = "HBEFA4/PC_petrol_Euro-6ab"
        f.write(f'    <vType id="disobedient_motorcycle" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="180" color=".2,.2,.8" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{sigma}" impatience="{impatience}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.4, 1.8)
        decel = np.random.uniform(3.3, 3.8)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="disobedient_truck" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="6" maxSpeed="100" color=".4,.1,.4" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{sigma}" impatience="{impatience}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.4)
        decel = np.random.uniform(3.0, 3.4)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="disobedient_truck/trailer" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="10.5" maxSpeed="100" color=".4,.4,.4" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{sigma}" impatience="{impatience}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')

        """ Electric vehicle types """
        emissionClass = "PHEMlight/zero" # ref: https://sumo.dlr.de/docs/Models/Emissions/PHEMlight.html
        accel = np.random.uniform(2.4, 2.8)
        decel = np.random.uniform(4.3, 4.7)
        length = np.random.triangular(3.7, 4.75, 5.7) * np.random.uniform(0.97, 1.07)
        f.write(f'    <vType id="electric_passenger" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="200" color="1,0,0" personNumber="{(int)(np.random.uniform(1, 5))}" sigma="{np.random.uniform(0.1, 0.5)}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.0, 3.4)
        decel = np.random.uniform(4.5, 5.0)
        length = np.random.triangular(3.7, 4.4, 5.0) * np.random.uniform(0.98, 1.02)
        f.write(f'    <vType id="electric_passenger/hatchback" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="240" color="0.8,0.8,0" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{np.random.uniform(0.1, 0.8)}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.5, 2.0)
        decel = np.random.uniform(3.0, 3.4)
        length = np.random.triangular(3.7, 4.5, 5.0) * np.random.uniform(0.98, 1.02)
        f.write(f'    <vType id="electric_passenger/van" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="130" color="1,1,1" personNumber="{(int)(np.random.uniform(1, 12))}" sigma="{np.random.uniform(0.1, 0.5)}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.5)
        decel = np.random.uniform(3.5, 4.1)
        length = np.random.triangular(12.0, 12.5, 18.0) * np.random.uniform(0.98, 1.02)
        f.write(f'    <vType id="electric_bus" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="100" color="1,.5,.5" personNumber="{(int)(np.random.uniform(1, 40))}" sigma="{np.random.uniform(0.1, 0.3)}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.5, 4.1)
        decel = np.random.uniform(5.0, 6.4)
        length = np.random.uniform(2.2, 2.6)
        f.write(f'    <vType id="electric_motorcycle" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="180" color=".2,.2,.8" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{np.random.uniform(0.1, 0.35)}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.4, 1.8)
        decel = np.random.uniform(3.3, 3.8)
        f.write(f'    <vType id="electric_truck" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="6" maxSpeed="100" color=".4,.1,.4" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{np.random.uniform(0.1, 0.35)}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.4)
        decel = np.random.uniform(3.0, 3.4)
        f.write(f'    <vType id="electric_truck/trailer" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="10.5" maxSpeed="100" color=".4,.4,.4" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{np.random.uniform(0.1, 0.3)}" speedFactor="1.2" speedDev="0.2" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')


        f.write('\n')
        f.write(f'    <route id="{route_id}" edges="{edges}"/>\n') # Replace {your_edges_here} with actual edges
        f.write('\n')

        # Write sorted flows to file
        for _, _, flow in flows:
            f.write(flow)

        f.write('</routes>\n')

    logging.info(f"Flow generation complete for model {model} id {idx}.")

