import numpy as np
import math
import logging
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

""" Route and other common flow attributes """
route_id = "r_0"
depart_lane = "free"
# "base": the vehicle is tried to be inserted at the position which lets its back be at the beginning of the lane (vehicle's front position=vehicle length)
depart_pos = "base"
# "avg": The average speed on the departure lane is used (or the minimum of 'speedLimit' and 'desired' if the lane is empty). If that speed is unsafe, departure is delayed.
depart_speed = "avg"
lanes = 3
car_following_model = 'IDM' # old: 'EIDM', 'Krauss' # https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#car-following_models
tau = 2.5 # reaction time: the desired time gap between vehicles in seconds

def get_electric_vehicles():
    return add_electric_vehicles
add_electric_vehicles = True

add_disobedient_vehicles = True

# Adjust the amplitude of the daily pattern
def adjust_amplitude(dly_pattern, amplitude_factor):
    return [round(value * math.exp(amplitude_factor * (value / max(dly_pattern)))) for value in dly_pattern]

def plot_vehicle_distributions():
    x_hours = np.arange(0, 24, 1)
    y_vehicles = bimodal_distribution_24h()
    print(y_vehicles)
    plt.figure(figsize=(10,6))
    plt.plot(x_hours, y_vehicles, marker='o')
    plt.title('Bimodal Distribution Over 24 Hours with Early Morning Curvature')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Vehicle Density')
    plt.xticks(x_hours)
    plt.grid(True)
    plt.show()

def bimodal_distribution_24h(amplitude):
    full_day_car_generation_base_demand = 250

    x = np.arange(0, 24, 1)  # 1-hour resolution
    # Nighttime pattern (curvature between 0:00 and 5:00)
    mu1 = 3.0 + np.random.uniform(-0.5, 0.5) # Nighttime peak around 3 AM
    sigma1 = 2 + np.random.uniform(-0.1, 0.1) # Wider nighttime peak
    A1 = np.random.uniform(0.0, 0.3)   # Lower amplitude for nighttime peak
    # Morning peak (sharp rise)
    mu2 = 8.5 + np.random.uniform(-0.5, 0.5)   # Morning peak around 7-8 AM
    sigma2 = 1.5 + np.random.uniform(-0.1, 0.1) # Narrower morning peak
    A2 = 1.0 + np.random.uniform(-0.03, 0.05)   # Higher amplitude for morning peak
    # Evening peak (broader)
    mu3 = 17.5 + np.random.uniform(-0.5, 0.5)   # Evening peak around 5-6 PM
    sigma3 = 2.5 + np.random.uniform(-0.2, 0.2) # Broader evening peak
    A3 = 0.9 + np.random.uniform(-0.05, 0.05)   # Slightly lower amplitude than morning
    # Midday plateau (lower amplitude)
    mu4 = 13.0 + np.random.uniform(-0.5, 0.5)   # Midday plateau around noon
    sigma4 = 3.5 + np.random.uniform(-0.3, 0.3) # Wider midday plateau
    A4 = 0.6 + np.random.uniform(-0.05, 0.05)   # Lower amplitude for midday plateau
    # Compute the bimodal distribution
    y = (
        A1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) +
        A2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2)) +
        A3 * np.exp(-((x - mu3) ** 2) / (2 * sigma3 ** 2)) + 
        A4 * np.exp(-((x - mu4) ** 2) / (2 * sigma4 ** 2))
    )
    y[0] = full_day_car_generation_base_demand / 100 * np.random.uniform(0.05, 0.15)
    # Add smoothed random noise to simulate natural fluctuations
    y = y * (1 + gaussian_filter(np.random.normal(0, 0.02, size=y.shape), sigma=1))
    # Normalize to [0, base_demand]
    y = y / np.max(y)
    y = y * full_day_car_generation_base_demand

    return adjust_amplitude(y.astype(int), amplitude)

# FIXME: this function doesn't work quite well. To be fixed
def triangular_distribution_24h(amplitude, x=np.arange(0, 24, 1)):
    full_day_car_generation_base_demand = 250

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
    y = y / np.max(y) * (full_day_car_generation_base_demand)
    # Ensure no negative values and set minimum values between (0, 50)
    y[y < 50] = np.random.uniform(0, 50)
    return adjust_amplitude(y.astype(int)[:24], amplitude)

def flow_generation(model, idx, daily_pattern, num_days):
    # Open a .rou.xml file to write flows
    with open(f"./traffic_environment/sumo/generated_flows_{model}_{idx}.rou.xml", "w") as f:
        edges = "seg_10_before seg_9_before seg_8_before seg_7_before seg_6_before seg_5_before seg_4_before seg_3_before seg_2_before seg_1_before seg_0_before seg_0_after seg_1_after"
        flows = [] # Collect flows here

        # Iterate over each pair of rates
        for day_index in range(num_days):
            for i in range(len(daily_pattern)):
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
                assert_tolerance = 0.5
                assert_allclose(total_distribution, 100, atol=assert_tolerance, err_msg=f"Total distribution is under {100-assert_tolerance}%")

                if add_disobedient_vehicles:
                    disobedient_normal_car_proportion = np.random.uniform(0.01, 0.1) * normal_car
                    disobedient_fast_car_proportion = np.random.uniform(0.5, 0.2) * fast_car
                    disobedient_van_proportion = np.random.uniform(0.01, 0.5) * van
                    disobedient_bus_proportion = np.random.uniform(0, 0.01) * bus
                    disobedient_motorcycle_proportion = np.random.uniform(0, 0.06) * motorcycle
                    disobedient_truck_proportion = np.random.uniform(0, 0.02) * truck
                    disobedient_trailer_proportion = np.random.uniform(0, 0.005) * trailer
                else:
                    disobedient_normal_car_proportion = 0
                    disobedient_fast_car_proportion = 0
                    disobedient_van_proportion = 0
                    disobedient_bus_proportion = 0
                    disobedient_motorcycle_proportion = 0
                    disobedient_truck_proportion = 0
                    disobedient_trailer_proportion = 0

                if add_electric_vehicles:
                    electric_normal_car_proportion = np.random.uniform(0.05, 0.1) * normal_car
                    electric_fast_car_proportion = np.random.uniform(0.07, 0.1) * fast_car
                    electric_van_proportion = np.random.uniform(0.02, 0.04) * van
                    electric_bus_proportion = np.random.uniform(0.005, 0.01) * bus
                    electric_motorcycle_proportion = np.random.uniform(0.03, 0.06) * motorcycle
                    electric_truck_proportion = np.random.uniform(0.01, 0.02) * truck
                    electric_trailer_proportion = np.random.uniform(0.0025, 0.005) * trailer
                else:
                    electric_normal_car_proportion = 0
                    electric_fast_car_proportion = 0
                    electric_van_proportion = 0
                    electric_bus_proportion = 0
                    electric_motorcycle_proportion = 0
                    electric_truck_proportion = 0
                    electric_trailer_proportion = 0

                proportions = {
                    "passenger": (normal_car-disobedient_normal_car_proportion-electric_normal_car_proportion),
                    "passenger/hatchback": (fast_car-disobedient_fast_car_proportion-electric_fast_car_proportion),
                    "passenger/van": (van-disobedient_van_proportion-electric_van_proportion),
                    "bus": (bus-disobedient_bus_proportion-electric_bus_proportion),
                    "motorcycle": (motorcycle-disobedient_motorcycle_proportion-electric_motorcycle_proportion),
                    "truck": (truck-disobedient_truck_proportion-electric_truck_proportion),
                    "truck/trailer": (trailer-disobedient_trailer_proportion-electric_trailer_proportion),

                    # Disobedient proportions:
                    "disobedient_passenger": disobedient_normal_car_proportion,
                    "disobedient_passenger/hatchback": disobedient_fast_car_proportion,
                    "disobedient_passenger/van": disobedient_van_proportion,
                    "disobedient_bus": disobedient_bus_proportion,
                    "disobedient_motorcycle": disobedient_motorcycle_proportion,
                    "disobedient_truck": disobedient_truck_proportion,
                    "disobedient_truck/trailer": disobedient_trailer_proportion,

                    # Electric proportions:
                    "electric_passenger": electric_normal_car_proportion,
                    "electric_passenger/hatchback": electric_fast_car_proportion,
                    "electric_passenger/van": electric_van_proportion,
                    "electric_bus": electric_bus_proportion,
                    "electric_motorcycle": electric_motorcycle_proportion,
                    "electric_truck": electric_truck_proportion,
                    "electric_truck/trailer": electric_trailer_proportion,
                }

                # Calculate start and end times for each flow
                # begin_time = (day_index * len(daily_pattern_ampl) * 1800) + (i * 1800)
                begin_time = day_index * len(daily_pattern) * 3600 + i * 3600
                end_time = begin_time + 3600
                                
                # Create flows for each vehicle type based on their proportions
                for vehicle_type in proportions:
                    vehs_gen = round(daily_pattern[i] * proportions[vehicle_type] / 100)
                    
                    if vehs_gen > 0:
                        if "disobedient" in vehicle_type and add_disobedient_vehicles:
                            flows.append((day_index, begin_time,
                                        f'    <flow id="{vehicle_type}_flow_{i}_day_{day_index}" type="{vehicle_type}" begin="{begin_time}" end="{end_time}" '
                                        f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                        f'route="{route_id}" vehsPerHour="{vehs_gen}" guiShape="{vehicle_type.removeprefix("disobedient_")}"/>\n'))
                        elif "electric" in vehicle_type and add_electric_vehicles:
                            flows.append((day_index, begin_time,
                                        f'    <flow id="{vehicle_type}_flow_{i}_day_{day_index}" type="{vehicle_type}" begin="{begin_time}" end="{end_time}" '
                                        f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                        f'route="{route_id}" vehsPerHour="{vehs_gen}" guiShape="{vehicle_type.removeprefix("electric_")}"/>\n'))
                        else:
                            flows.append((day_index, begin_time,
                                        f'    <flow id="{vehicle_type}_flow_{i}_day_{day_index}" type="{vehicle_type}" begin="{begin_time}" end="{end_time}" '
                                        f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                        f'route="{route_id}" vehsPerHour="{vehs_gen}" guiShape="{vehicle_type}"/>\n'))

        # Sort flows by begin time only (no need to sort by day index first)
        flows.sort(key=lambda x: (x[1]))

        # Write sorted flows
        f.write('<routes>\n')
        f.write('\n')
        
        """ Define vehicle types """
        # Define specific vehicle types
        accel = np.random.uniform(2.4, 2.8)
        decel = np.random.uniform(4.3, 4.7)
        length = np.random.triangular(3.7, 4.75, 5.7) * np.random.uniform(0.97, 1.07)
        emissionClass = "HBEFA4/PC_petrol_Euro-4"
        f.write(f'    <vType id="passenger" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.5)}" length="{length}" maxSpeed="{180/3.6}" color="1,0,0" personNumber="{(int)(np.random.uniform(1, 5))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.0, 3.4)
        decel = np.random.uniform(4.5, 5.0)
        length = np.random.triangular(3.7, 4.4, 5.0) * np.random.uniform(0.98, 1.02)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="passenger/hatchback" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.7)}" length="{length}" maxSpeed="{240/3.6}" color="0.8,0.8,0" personNumber="{(int)(np.random.uniform(1, 3))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.5, 2.0)
        decel = np.random.uniform(3.0, 3.4)
        length = np.random.triangular(3.7, 4.5, 5.0) * np.random.uniform(0.98, 1.02)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="passenger/van" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.5)}" length="{length}" maxSpeed="{130/3.6}" color="1,1,1" personNumber="{(int)(np.random.uniform(1, 12))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.5)
        decel = np.random.uniform(3.5, 4.1)
        length = np.random.triangular(12.0, 12.5, 18.0) * np.random.uniform(0.98, 1.02)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="bus" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.3)}" length="{length}" maxSpeed="{100/3.6}" color="1,.5,.5" personNumber="{(int)(np.random.uniform(1, 40))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.5, 4.1)
        decel = np.random.uniform(5.0, 6.4)
        length = np.random.uniform(2.2, 2.6)
        emissionClass = "HBEFA4/PC_petrol_Euro-6ab" # UBus_Electric_Artic_gt18t
        f.write(f'    <vType id="motorcycle" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.35)}" length="{length}" maxSpeed="{180/3.6}" color=".2,.2,.8" personNumber="{(int)(np.random.uniform(1, 2))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.4, 1.8)
        decel = np.random.uniform(3.3, 3.8)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="truck" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.35)}" length="8" maxSpeed="{100/3.6}" color=".4,.1,.4" personNumber="{(int)(np.random.uniform(1, 3))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.4)
        decel = np.random.uniform(3.0, 3.4)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="truck/trailer" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.3)}" length="13.6" maxSpeed="{100/3.6}" color=".4,.4,.4" personNumber="{(int)(np.random.uniform(1, 2))}" emissionClass="{emissionClass}"/>\n')

        """ Disobedient vehicle types """
        accel = np.random.uniform(2.4, 2.8)
        decel = np.random.uniform(4.3, 4.7)
        length = np.random.triangular(3.7, 4.75, 5.7) * np.random.uniform(0.97, 1.07)
        impatience = np.random.uniform(0.5, 1)
        sigma = np.random.uniform(0.5, 1) # Slight deviations from the desired speed or Less precise reactions to traffic conditions (e.g., following distances, acceleration/deceleration)
        speedFactor = np.random.uniform(1, 1.5) # between 0 and 50% faster. This allows the vehicle to exceed the lane's max speed.
        speedDev = np.random.uniform(0, 0.1) # between 0 and 10% deviation. Introduces some randomness in how much faster or slower than the calculated max speed (lane max speed * speedFactor) the vehicle might drive
        emissionClass = "HBEFA4/PC_petrol_Euro-4"
        f.write(f'    <vType id="disobedient_passenger" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{200/3.6}" color="1,0,0" personNumber="{(int)(np.random.uniform(1, 5))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.0, 3.4)
        decel = np.random.uniform(4.5, 5.0)
        length = np.random.triangular(3.7, 4.4, 5.0) * np.random.uniform(0.98, 1.02)
        impatience = np.random.uniform(0.75, 1)
        sigma=np.random.uniform(0.85, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="disobedient_passenger/hatchback" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{240/3.6}" color="0.8,0.8,0" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.5, 2.0)
        decel = np.random.uniform(3.0, 3.4)
        length = np.random.triangular(3.7, 4.5, 5.0) * np.random.uniform(0.98, 1.02)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="disobedient_passenger/van" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{130/3.6}" color="1,1,1" personNumber="{(int)(np.random.uniform(1, 12))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.5)
        decel = np.random.uniform(3.5, 4.1)
        length = np.random.triangular(12.0, 12.5, 18.0) * np.random.uniform(0.98, 1.02)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="disobedient_bus" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{100/3.6}" color="1,.5,.5" personNumber="{(int)(np.random.uniform(1, 40))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.5, 4.1)
        decel = np.random.uniform(5.0, 6.4)
        length = np.random.uniform(2.2, 2.6)
        impatience = np.random.uniform(0.75, 1)
        sigma=np.random.uniform(0.75, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/PC_petrol_Euro-6ab"
        f.write(f'    <vType id="disobedient_motorcycle" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{180/3.6}" color=".2,.2,.8" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.4, 1.8)
        decel = np.random.uniform(3.3, 3.8)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="disobedient_truck" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="6" maxSpeed="{100/3.6}" color=".4,.1,.4" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.4)
        decel = np.random.uniform(3.0, 3.4)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="disobedient_truck/trailer" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="10.5" maxSpeed="{100/3.6}" color=".4,.4,.4" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')

        """ Electric vehicle types """
        speedFactor = 1
        speedDev = 0
        emissionClass = "PHEMlight/zero" # ref: https://sumo.dlr.de/docs/Models/Emissions/PHEMlight.html
        accel = np.random.uniform(2.4, 2.8)
        decel = np.random.uniform(4.3, 4.7)
        length = np.random.triangular(3.7, 4.75, 5.7) * np.random.uniform(0.97, 1.07)
        f.write(f'    <vType id="electric_passenger" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{200/3.6}" color="1,0,0" personNumber="{(int)(np.random.uniform(1, 5))}" sigma="{np.random.uniform(0.1, 0.5)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.0, 3.4)
        decel = np.random.uniform(4.5, 5.0)
        length = np.random.triangular(3.7, 4.4, 5.0) * np.random.uniform(0.98, 1.02)
        f.write(f'    <vType id="electric_passenger/hatchback" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{240/3.6}" color="0.8,0.8,0" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{np.random.uniform(0.1, 0.8)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.5, 2.0)
        decel = np.random.uniform(3.0, 3.4)
        length = np.random.triangular(3.7, 4.5, 5.0) * np.random.uniform(0.98, 1.02)
        f.write(f'    <vType id="electric_passenger/van" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{130/3.6}" color="1,1,1" personNumber="{(int)(np.random.uniform(1, 12))}" sigma="{np.random.uniform(0.1, 0.5)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.5)
        decel = np.random.uniform(3.5, 4.1)
        length = np.random.triangular(12.0, 12.5, 18.0) * np.random.uniform(0.98, 1.02)
        f.write(f'    <vType id="electric_bus" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{100/3.6}" color="1,.5,.5" personNumber="{(int)(np.random.uniform(1, 40))}" sigma="{np.random.uniform(0.1, 0.3)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.5, 4.1)
        decel = np.random.uniform(5.0, 6.4)
        length = np.random.uniform(2.2, 2.6)
        f.write(f'    <vType id="electric_motorcycle" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{180/3.6}" color=".2,.2,.8" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{np.random.uniform(0.1, 0.35)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.4, 1.8)
        decel = np.random.uniform(3.3, 3.8)
        f.write(f'    <vType id="electric_truck" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="6" maxSpeed="{100/3.6}" color=".4,.1,.4" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{np.random.uniform(0.1, 0.35)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.4)
        decel = np.random.uniform(3.0, 3.4)
        f.write(f'    <vType id="electric_truck/trailer" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="10.5" maxSpeed="{100/3.6}" color=".4,.4,.4" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{np.random.uniform(0.1, 0.3)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')


        f.write('\n')
        f.write(f'    <route id="{route_id}" edges="{edges}"/>\n') # Replace {your_edges_here} with actual edges
        f.write('\n')

        # Write sorted flows to file
        for _, _, flow in flows:
            f.write(flow)

        f.write('</routes>\n')

    logging.info(f"Flow generation complete for model {model} id {idx}.")

def flow_generation_fix_num_veh(model, idx, base_num_veh_per_hr, num_of_hrs, num_of_episodes, num_of_intervals, op_mode):
    # Open a .rou.xml file to write flows
    with open(f"./traffic_environment/sumo/generated_flows_{model}_{idx}.rou.xml", "w") as f:
        edges = "seg_10_before seg_9_before seg_8_before seg_7_before seg_6_before seg_5_before seg_4_before seg_3_before seg_2_before seg_1_before seg_0_before seg_0_after seg_1_after"
        flows = [] # Collect flows here

        # Iterate over each pair of rates
        for ep in range(num_of_episodes):
            for i in range(num_of_hrs * num_of_intervals):
                num_veh_per_hr_temp = base_num_veh_per_hr + (i // num_of_hrs) * 100 if op_mode == "train" else base_num_veh_per_hr
                # Vehicle type distributions
                trucks = np.random.uniform(10, 15)
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
                assert_tolerance = 0.5
                assert_allclose(total_distribution, 100, atol=assert_tolerance, err_msg=f"Total distribution is under {100-assert_tolerance}%")

                if add_disobedient_vehicles:
                    disobedient_normal_car_proportion = np.random.uniform(0.01, 0.1) * normal_car
                    disobedient_fast_car_proportion = np.random.uniform(0.5, 0.2) * fast_car
                    disobedient_van_proportion = np.random.uniform(0.01, 0.5) * van
                    disobedient_bus_proportion = np.random.uniform(0, 0.01) * bus
                    disobedient_motorcycle_proportion = np.random.uniform(0, 0.06) * motorcycle
                    disobedient_truck_proportion = np.random.uniform(0, 0.02) * truck
                    disobedient_trailer_proportion = np.random.uniform(0, 0.005) * trailer
                else:
                    disobedient_normal_car_proportion = 0
                    disobedient_fast_car_proportion = 0
                    disobedient_van_proportion = 0
                    disobedient_bus_proportion = 0
                    disobedient_motorcycle_proportion = 0
                    disobedient_truck_proportion = 0
                    disobedient_trailer_proportion = 0

                if add_electric_vehicles:
                    electric_normal_car_proportion = np.random.uniform(0.05, 0.1) * normal_car
                    electric_fast_car_proportion = np.random.uniform(0.07, 0.1) * fast_car
                    electric_van_proportion = np.random.uniform(0.02, 0.04) * van
                    electric_bus_proportion = np.random.uniform(0.005, 0.01) * bus
                    electric_motorcycle_proportion = np.random.uniform(0.03, 0.06) * motorcycle
                    electric_truck_proportion = np.random.uniform(0.01, 0.02) * truck
                    electric_trailer_proportion = np.random.uniform(0.0025, 0.005) * trailer
                else:
                    electric_normal_car_proportion = 0
                    electric_fast_car_proportion = 0
                    electric_van_proportion = 0
                    electric_bus_proportion = 0
                    electric_motorcycle_proportion = 0
                    electric_truck_proportion = 0
                    electric_trailer_proportion = 0

                proportions = {
                    "passenger": (normal_car-disobedient_normal_car_proportion-electric_normal_car_proportion),
                    "passenger/hatchback": (fast_car-disobedient_fast_car_proportion-electric_fast_car_proportion),
                    "passenger/van": (van-disobedient_van_proportion-electric_van_proportion),
                    "bus": (bus-disobedient_bus_proportion-electric_bus_proportion),
                    "motorcycle": (motorcycle-disobedient_motorcycle_proportion-electric_motorcycle_proportion),
                    "truck": (truck-disobedient_truck_proportion-electric_truck_proportion),
                    "truck/trailer": (trailer-disobedient_trailer_proportion-electric_trailer_proportion),

                    # Disobedient proportions:
                    "disobedient_passenger": disobedient_normal_car_proportion,
                    "disobedient_passenger/hatchback": disobedient_fast_car_proportion,
                    "disobedient_passenger/van": disobedient_van_proportion,
                    "disobedient_bus": disobedient_bus_proportion,
                    "disobedient_motorcycle": disobedient_motorcycle_proportion,
                    "disobedient_truck": disobedient_truck_proportion,
                    "disobedient_truck/trailer": disobedient_trailer_proportion,

                    # Electric proportions:
                    "electric_passenger": electric_normal_car_proportion,
                    "electric_passenger/hatchback": electric_fast_car_proportion,
                    "electric_passenger/van": electric_van_proportion,
                    "electric_bus": electric_bus_proportion,
                    "electric_motorcycle": electric_motorcycle_proportion,
                    "electric_truck": electric_truck_proportion,
                    "electric_truck/trailer": electric_trailer_proportion,
                }

                # Calculate start and end times for each flow
                begin_time = i * 3600 + (num_of_hrs * num_of_intervals * ep * 3600)
                end_time = begin_time + 3600

                # Create flows for each vehicle type based on their proportions
                for vehicle_type in proportions:
                    vehs_gen = round(num_veh_per_hr_temp * proportions[vehicle_type] / 100)
                    
                    if vehs_gen > 0:
                        if "disobedient" in vehicle_type and add_disobedient_vehicles:
                            flows.append((begin_time,
                                        f'    <flow id="{vehicle_type}_flow_{i}_ep_{ep}" type="{vehicle_type}" begin="{begin_time}" end="{end_time}" '
                                        f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                        f'route="{route_id}" vehsPerHour="{vehs_gen}" guiShape="{vehicle_type.removeprefix("disobedient_")}"/>\n'))
                        elif "electric" in vehicle_type and add_electric_vehicles:
                            flows.append((begin_time,
                                        f'    <flow id="{vehicle_type}_flow_{i}_ep_{ep}" type="{vehicle_type}" begin="{begin_time}" end="{end_time}" '
                                        f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                        f'route="{route_id}" vehsPerHour="{vehs_gen}" guiShape="{vehicle_type.removeprefix("electric_")}"/>\n'))
                        else:
                            flows.append((begin_time,
                                        f'    <flow id="{vehicle_type}_flow_{i}_ep_{ep}" type="{vehicle_type}" begin="{begin_time}" end="{end_time}" '
                                        f'departLane="{depart_lane}" departPos="{depart_pos}" departSpeed="{depart_speed}" '
                                        f'route="{route_id}" vehsPerHour="{vehs_gen}" guiShape="{vehicle_type}"/>\n'))

        # Sort flows by begin time only (no need to sort by day index first)
        flows.sort(key=lambda x: (x[0]))

        # Write sorted flows
        f.write('<routes>\n')
        f.write('\n')
        
        """ Define vehicle types """
        # Define specific vehicle types
        accel = np.random.uniform(2.4, 2.8)
        decel = np.random.uniform(4.3, 4.7)
        length = np.random.triangular(3.7, 4.75, 5.7) * np.random.uniform(0.97, 1.07)
        emissionClass = "HBEFA4/PC_petrol_Euro-4"
        f.write(f'    <vType id="passenger" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.5)}" length="{length}" maxSpeed="{180/3.6}" color="1,0,0" personNumber="{(int)(np.random.uniform(1, 5))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.0, 3.4)
        decel = np.random.uniform(4.5, 5.0)
        length = np.random.triangular(3.7, 4.4, 5.0) * np.random.uniform(0.98, 1.02)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="passenger/hatchback" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.7)}" length="{length}" maxSpeed="{240/3.6}" color="0.8,0.8,0" personNumber="{(int)(np.random.uniform(1, 3))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.5, 2.0)
        decel = np.random.uniform(3.0, 3.4)
        length = np.random.triangular(3.7, 4.5, 5.0) * np.random.uniform(0.98, 1.02)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="passenger/van" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.5)}" length="{length}" maxSpeed="{130/3.6}" color="1,1,1" personNumber="{(int)(np.random.uniform(1, 12))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.5)
        decel = np.random.uniform(3.5, 4.1)
        length = np.random.triangular(12.0, 12.5, 18.0) * np.random.uniform(0.98, 1.02)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="bus" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.3)}" length="{length}" maxSpeed="{100/3.6}" color="1,.5,.5" personNumber="{(int)(np.random.uniform(1, 40))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.5, 4.1)
        decel = np.random.uniform(5.0, 6.4)
        length = np.random.uniform(2.2, 2.6)
        emissionClass = "HBEFA4/PC_petrol_Euro-6ab" # UBus_Electric_Artic_gt18t
        f.write(f'    <vType id="motorcycle" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.35)}" length="{length}" maxSpeed="{180/3.6}" color=".2,.2,.8" personNumber="{(int)(np.random.uniform(1, 2))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.4, 1.8)
        decel = np.random.uniform(3.3, 3.8)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="truck" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.35)}" length="8" maxSpeed="{100/3.6}" color=".4,.1,.4" personNumber="{(int)(np.random.uniform(1, 3))}" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.4)
        decel = np.random.uniform(3.0, 3.4)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="truck/trailer" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" sigma="{np.random.uniform(0.1, 0.3)}" length="13.6" maxSpeed="{100/3.6}" color=".4,.4,.4" personNumber="{(int)(np.random.uniform(1, 2))}" emissionClass="{emissionClass}"/>\n')

        """ Disobedient vehicle types """
        accel = np.random.uniform(2.4, 2.8)
        decel = np.random.uniform(4.3, 4.7)
        length = np.random.triangular(3.7, 4.75, 5.7) * np.random.uniform(0.97, 1.07)
        impatience = np.random.uniform(0.5, 1)
        sigma = np.random.uniform(0.5, 1) # Slight deviations from the desired speed or Less precise reactions to traffic conditions (e.g., following distances, acceleration/deceleration)
        speedFactor = np.random.uniform(1, 1.5) # between 0 and 50% faster. This allows the vehicle to exceed the lane's max speed.
        speedDev = np.random.uniform(0, 0.1) # between 0 and 10% deviation. Introduces some randomness in how much faster or slower than the calculated max speed (lane max speed * speedFactor) the vehicle might drive
        emissionClass = "HBEFA4/PC_petrol_Euro-4"
        f.write(f'    <vType id="disobedient_passenger" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{200/3.6}" color="1,0,0" personNumber="{(int)(np.random.uniform(1, 5))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.0, 3.4)
        decel = np.random.uniform(4.5, 5.0)
        length = np.random.triangular(3.7, 4.4, 5.0) * np.random.uniform(0.98, 1.02)
        impatience = np.random.uniform(0.75, 1)
        sigma=np.random.uniform(0.85, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="disobedient_passenger/hatchback" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{240/3.6}" color="0.8,0.8,0" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.5, 2.0)
        decel = np.random.uniform(3.0, 3.4)
        length = np.random.triangular(3.7, 4.5, 5.0) * np.random.uniform(0.98, 1.02)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/PC_petrol_Euro-5"
        f.write(f'    <vType id="disobedient_passenger/van" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{130/3.6}" color="1,1,1" personNumber="{(int)(np.random.uniform(1, 12))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.5)
        decel = np.random.uniform(3.5, 4.1)
        length = np.random.triangular(12.0, 12.5, 18.0) * np.random.uniform(0.98, 1.02)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="disobedient_bus" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{100/3.6}" color="1,.5,.5" personNumber="{(int)(np.random.uniform(1, 40))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.5, 4.1)
        decel = np.random.uniform(5.0, 6.4)
        length = np.random.uniform(2.2, 2.6)
        impatience = np.random.uniform(0.75, 1)
        sigma=np.random.uniform(0.75, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/PC_petrol_Euro-6ab"
        f.write(f'    <vType id="disobedient_motorcycle" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{180/3.6}" color=".2,.2,.8" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.4, 1.8)
        decel = np.random.uniform(3.3, 3.8)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="disobedient_truck" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="6" maxSpeed="{100/3.6}" color=".4,.1,.4" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.4)
        decel = np.random.uniform(3.0, 3.4)
        impatience = np.random.uniform(0.5, 1)
        sigma=np.random.uniform(0.5, 1)
        speedFactor = np.random.uniform(1, 1.5)
        speedDev = np.random.uniform(0, 0.1)
        emissionClass = "HBEFA4/RT_le7.5t_Euro-VI_A-C"
        f.write(f'    <vType id="disobedient_truck/trailer" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="10.5" maxSpeed="{100/3.6}" color=".4,.4,.4" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{sigma}" impatience="{impatience}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')

        """ Electric vehicle types """
        speedFactor = 1
        speedDev = 0
        emissionClass = "PHEMlight/zero" # ref: https://sumo.dlr.de/docs/Models/Emissions/PHEMlight.html
        accel = np.random.uniform(2.4, 2.8)
        decel = np.random.uniform(4.3, 4.7)
        length = np.random.triangular(3.7, 4.75, 5.7) * np.random.uniform(0.97, 1.07)
        f.write(f'    <vType id="electric_passenger" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{200/3.6}" color="1,0,0" personNumber="{(int)(np.random.uniform(1, 5))}" sigma="{np.random.uniform(0.1, 0.5)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.0, 3.4)
        decel = np.random.uniform(4.5, 5.0)
        length = np.random.triangular(3.7, 4.4, 5.0) * np.random.uniform(0.98, 1.02)
        f.write(f'    <vType id="electric_passenger/hatchback" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{240/3.6}" color="0.8,0.8,0" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{np.random.uniform(0.1, 0.8)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.5, 2.0)
        decel = np.random.uniform(3.0, 3.4)
        length = np.random.triangular(3.7, 4.5, 5.0) * np.random.uniform(0.98, 1.02)
        f.write(f'    <vType id="electric_passenger/van" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{130/3.6}" color="1,1,1" personNumber="{(int)(np.random.uniform(1, 12))}" sigma="{np.random.uniform(0.1, 0.5)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.5)
        decel = np.random.uniform(3.5, 4.1)
        length = np.random.triangular(12.0, 12.5, 18.0) * np.random.uniform(0.98, 1.02)
        f.write(f'    <vType id="electric_bus" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{100/3.6}" color="1,.5,.5" personNumber="{(int)(np.random.uniform(1, 40))}" sigma="{np.random.uniform(0.1, 0.3)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(3.5, 4.1)
        decel = np.random.uniform(5.0, 6.4)
        length = np.random.uniform(2.2, 2.6)
        f.write(f'    <vType id="electric_motorcycle" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="{length}" maxSpeed="{180/3.6}" color=".2,.2,.8" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{np.random.uniform(0.1, 0.35)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.4, 1.8)
        decel = np.random.uniform(3.3, 3.8)
        f.write(f'    <vType id="electric_truck" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="6" maxSpeed="{100/3.6}" color=".4,.1,.4" personNumber="{(int)(np.random.uniform(1, 3))}" sigma="{np.random.uniform(0.1, 0.35)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')
        accel = np.random.uniform(1.0, 1.4)
        decel = np.random.uniform(3.0, 3.4)
        f.write(f'    <vType id="electric_truck/trailer" carFollowModel= "{car_following_model}" accel="{accel}" decel="{decel}" length="10.5" maxSpeed="{100/3.6}" color=".4,.4,.4" personNumber="{(int)(np.random.uniform(1, 2))}" sigma="{np.random.uniform(0.1, 0.3)}" speedFactor="{speedFactor}" speedDev="{speedDev}" lcStrategic="0.5" lcCooperative="0.1" lcSpeedGain="1.5" emissionClass="{emissionClass}"/>\n')


        f.write('\n')
        f.write(f'    <route id="{route_id}" edges="{edges}"/>\n') # Replace {your_edges_here} with actual edges
        f.write('\n')

        # Write sorted flows to file
        for _, flow in flows:
            f.write(flow)

        f.write('</routes>\n')

    logging.info(f"Flow generation complete for model {model} id {idx}.")

if __name__ == '__main__':
    # flow_generation_fix_num_veh("DQN", 0, 250, 8)
    # flow_generation_fix_num_veh("DQN", num_envs_per_model + 1, 500, episode_length)
    logging.info("Flow generation ran individually")