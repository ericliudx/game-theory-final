from agent import CommuterAgent
import pandas as pd
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from scipy.optimize import root_scalar


class TransportModel(Model):
    def __init__(
        self, 
        num_agents=20000, 
        fare_discount=0.0, 
        car_toll=0.0, 
        car_cost = 5 * 2,
        bus_cost = 2.0 * 2,
        train_cost = 3.0 * 2,
        lambda_public = 0.8,
        lambda_private = 0.6,
        road_capacity=10000, 
        median_income=60000,
        mean_income=88000,
        avg_freeflow_duration = 1.0,
        gas_cost=0,
        diesel_cost=0,
        rush_hours = 3,
        car_hourly_value = 23.12,
        truck_hourly_value = 64.48,
        truck_ratio = 1.0/12.0,
        train_cost_base = 11.0 * 2.0,
        bus_cost_base = 5.0 * 2.0,
        train_baseline_ridership = 0.125,
        bus_baseline_ridership = 0.11,
        car_enforcement_pct = 0.1,
        road_maintainence_cost_car = 1.0,
        road_maintainence_cost_truck = 10.0,
        commute_distance_mean = 2.3,
        commute_distance_sigma = 0.5,
        width=10, 
        height=10, 
        seed = None
    ):
        
        super().__init__(seed=seed)
        self.num_agents = num_agents
        self.initial_car_toll = 9.2               # initial toll, e.g., 0.0
        self.initial_fare_discount = 0.0               # always zero at start
        self.car_toll = self.initial_car_toll
        self.fare_discount = self.initial_fare_discount

        self.policy_step = 12                          # month to apply new policy
        self.new_car_toll = car_toll                   # set this externally later
        self.new_fare_discount = fare_discount                   # set this externally later
        self.width = width
        self.height = height

        self.car_cost = car_cost
        self.bus_cost = bus_cost
        self.train_cost = train_cost

        self.lambda_private = lambda_private
        self.lambda_public = lambda_public

        self.road_capacity = road_capacity
        self.congestion_level = 1.3
        self.v_over_c = ((self.congestion_level - 1) / 0.15) ** (1/4)
        self.road_capacity = (self.num_agents * 0.5) / self.v_over_c
        self.median_income = median_income
        self.mean_income = mean_income
        
        #mode counts and percentages
        self.classes = ["upper", "middle", "lower"]
        self.mode_counts = {
            "car": {
                "upper": 0,
                "middle": 0,
                "lower": 0,
                },
            "bus": {
                "upper": 0,
                "middle": 0,
                "lower": 0,
                },
            "train": {
                "upper": 0,
                "middle": 0,
                "lower": 0,
                },
            "bike_walk": {
                "upper": 0,
                "middle": 0,
                "lower": 0,
                },
        }
        self.total_mode_counts = {
            "car": 0,
            "bus": 0,
            "train": 0,
            "bike_walk": 0,
        }
        
        self.car_share_pct = 0
        self.bus_share_pct = 0 
        self.train_share_pct = 0
        self.bike_walk_share_pct = 0
        
        self.car_share_pct_upper = 0
        self.bus_share_pct_upper = 0 
        self.train_share_pct_upper = 0
        self.bike_walk_share_pct_upper = 0
        self.car_share_pct_middle = 0
        self.bus_share_pct_middle = 0 
        self.train_share_pct_middle = 0
        self.bike_walk_share_pct_middle = 0
        self.car_share_pct_lower = 0
        self.bus_share_pct_lower = 0 
        self.train_share_pct_lower = 0
        self.bike_walk_share_pct_lower = 0

        #incomes
        self.socio_groups = []
        def objective(alpha):
            xm = mean_income * (alpha - 1) / alpha
            return xm * 2**(1 / alpha) - median_income

        result = root_scalar(objective, bracket=[1.01, 10], method='brentq')
        self.alpha = result.root
        xm = mean_income * (self.alpha - 1) / self.alpha
        self.incomes = xm * (1 + np.random.pareto(self.alpha, size=self.num_agents))

        self.car_ownerships = []
        self.sensitivities = []
        self.time_values = []
        
        self.upper = 0
        self.middle = 0
        self.lower = 0
        
        self.car_owners = 0
        
        self.avg_freeflow_duration = avg_freeflow_duration
        self.total_freeflow_hours = 0
        self.car_congestion_hours = 0.0
        self.truck_congestion_hours = 0
        self.gas_cost = gas_cost
        self.diesel_cost = diesel_cost
        self.car_hourly_value = car_hourly_value
        self.truck_hourly_value = truck_hourly_value
        self.rush_hours = rush_hours
        self.gasoline_consumption_hr = 0.25
        self.diesel_consumption_hr = 0.65
        self.car_cong_ghg_hourly = 2.22
        self.truck_cong_ghg_hourly = 6.64
        self.car_freeflow_ghg_hourly = 16.0
        self.truck_ratio = truck_ratio
        
        self.total_car_cong_cost = 0
        self.total_truck_cong_cost = 0
        self.total_cong_cost = 0
        
        self.total_cong_ghg = 0
        self.total_car_cong_ghg = 0
        self.total_truck_cong_ghg = 0
        self.total_car_freeflow_ghg = 0
        self.total_ghg = 0
        self.total_ghg_sum = 0
        
        self.train_rev = 0
        self.train_cost_base = train_cost_base
        self.train_baseline_ridership = train_baseline_ridership
        self.train_cost_discount = 0
        self.train_cost_per_rider = 0
        self.bus_rev = 0
        self.bus_cost_base = bus_cost_base
        self.bus_cost_per_rider = 0
        self.bus_baseline_ridership = bus_baseline_ridership
        self.bus_cost_discount = 0
        self.total_transit_rev = 0
        self.total_transit_cost = 0
        self.total_transit_profit = 0
        
        self.car_enforcement_pct = car_enforcement_pct
        self.road_maintainence_cost_car = road_maintainence_cost_car
        self.road_maintainence_cost_truck = road_maintainence_cost_truck
        self.road_maintainence = 0
        self.toll_revenue = 0
        self.toll_profit = 0
        self.total_system_profit = 0
        
        self.commute_distance_mean = commute_distance_mean
        self.commute_distance_sigma = commute_distance_sigma
        self.distances = []
        for income in self.incomes:
            self.sensitivities.append(self.median_income/income)
            hourly_wage = income / 2080
            self.time_values.append(0.5*hourly_wage)
            prob = 1 / (1 + np.exp(-(-10 + 0.0002 * income)))
            random_draw = np.random.rand(*np.shape(prob))
            ownership = random_draw < prob
            if(ownership):
                self.car_owners+=1
            self.car_ownerships.append(ownership)
            if(income < 0.75 * self.median_income):
                self.socio_groups.append("lower")
                self.distances.append(np.random.lognormal(mean=2.0, sigma=0.4))
                self.lower+=1
            elif(income > 2 * self.median_income):
                self.socio_groups.append("upper")
                self.distances.append(np.random.lognormal(mean=2.3, sigma=0.5))
                self.upper+=1
            else:
                self.socio_groups.append("middle")
                self.distances.append(np.random.lognormal(mean=2.6, sigma=0.5))
                self.middle+=1
            
        # Batch-create agents
        CommuterAgent.create_agents(
            model=self,
            n=num_agents,
            socio_group=self.socio_groups,
            income=self.incomes,
            car_owner=self.car_ownerships,
            price_sensitivity=self.sensitivities,
            distance = self.distances,
            time_value = self.time_values,
        )

        self.datacollector = DataCollector(
            model_reporters={
                "congestion_level": "congestion_level",
                "car_share_pct": "car_share_pct",
                "bus_share_pct": "bus_share_pct",
                "train_share_pct": "train_share_pct",
                "bike_walk_share_pct": "bike_walk_share_pct",
                
                "car_share_pct_upper": "car_share_pct_upper",
                "bus_share_pct_upper": "bus_share_pct_upper",
                "train_share_pct_upper": "train_share_pct_upper",
                "bike_walk_share_pct_upper": "bike_walk_share_pct_upper",
                "car_share_pct_middle": "car_share_pct_middle",
                "bus_share_pct_middle": "bus_share_pct_middle",
                "train_share_pct_middle": "train_share_pct_middle",
                "bike_walk_share_pct_middle": "bike_walk_share_pct_middle",
                "car_share_pct_lower": "car_share_pct_lower",
                "bus_share_pct_lower": "bus_share_pct_lower",
                "train_share_pct_lower": "train_share_pct_lower",
                "bike_walk_share_pct_lower": "bike_walk_share_pct_lower",
                "total_ghg": "total_ghg",
                "total_ghg_sum": "total_ghg_sum",
                "total_system_profit": "total_system_profit",
            },
            agent_reporters={
                "mode_choice": "mode_choice",
            },
        )


        self.running = True
        # self.datacollector.collect(self)
        
    def step(self):
        self.agents.shuffle_do("step")
        self.update_congestion()
        self.total_bike_walk_count = self.mode_counts["bike_walk"]["upper"]+self.mode_counts["bike_walk"]["middle"]+self.mode_counts["bike_walk"]["lower"]
        self.mode_share_pcts()
        self.congestion_costs()
        self.ghg_emissions()
        self.transit_costs()
        self.car_road_costs()
        self.total_system_profit = self.toll_profit + self.total_transit_profit - self.total_cong_cost
        if self.steps == self.policy_step:
            self.car_toll += self.new_car_toll
            self.fare_discount = self.new_fare_discount
        self.datacollector.collect(self)

    def update_congestion(self):
        cars = sum(1 for a in self.agents if a.mode_choice == 'car')
        self.v_over_c = cars / self.road_capacity
        self.congestion_level = 1 + 0.15 * (self.v_over_c) ** 4
    def mode_share_pcts(self):
        self.car_share_pct = (self.total_mode_counts["car"]/ self.num_agents) * 100
        self.bus_share_pct = (self.total_mode_counts["bus"]  / self.num_agents) * 100
        self.train_share_pct = (self.total_mode_counts["train"] / self.num_agents) * 100
        self.bike_walk_share_pct = (self.total_mode_counts["bike_walk"] / self.num_agents) * 100
        
        self.car_share_pct_upper = (self.mode_counts["car"]["upper"]/ self.upper) * 100
        self.bus_share_pct_upper = (self.mode_counts["bus"]["upper"]  / self.upper) * 100
        self.train_share_pct_upper = (self.mode_counts["train"]["upper"] / self.upper) * 100
        self.bike_walk_share_pct_upper = (self.mode_counts["bike_walk"]["upper"] / self.upper) * 100
        
        self.car_share_pct_middle = (self.mode_counts["car"]["middle"]/ self.middle) * 100
        self.bus_share_pct_middle = (self.mode_counts["bus"]["middle"]  / self.middle) * 100
        self.train_share_pct_middle = (self.mode_counts["train"]["middle"] / self.middle) * 100
        self.bike_walk_share_pct_middle = (self.mode_counts["bike_walk"]["middle"] / self.middle) * 100
        
        self.car_share_pct_lower = (self.mode_counts["car"]["lower"]/ self.lower) * 100
        self.bus_share_pct_lower = (self.mode_counts["bus"]["lower"]  / self.lower) * 100
        self.train_share_pct_lower = (self.mode_counts["train"]["lower"] / self.lower) * 100
        self.bike_walk_share_pct_lower = (self.mode_counts["bike_walk"]["lower"] / self.lower) * 100
    def congestion_costs(self):
        self.total_freeflow_hours = self.total_mode_counts["car"] * self.avg_freeflow_duration
        self.car_congestion_hours = (self.total_mode_counts["car"] * self.avg_freeflow_duration) * (self.congestion_level - 1.0)
        self.total_car_cong_cost = self.car_congestion_hours * (self.gasoline_consumption_hr * self.gas_cost + self.car_hourly_value)
        self.truck_congestion_hours = self.truck_ratio * self.num_agents * self.rush_hours * (self.congestion_level - 1)
        self.total_truck_cong_cost = self.truck_congestion_hours * (self.diesel_consumption_hr * self.diesel_cost + self.truck_hourly_value)
        self.total_cong_cost = self.total_car_cong_cost + self.total_truck_cong_cost
    def ghg_emissions(self):
        self.total_car_cong_ghg = self.car_congestion_hours * self.car_cong_ghg_hourly
        self.total_truck_cong_ghg = self.truck_congestion_hours * self.truck_cong_ghg_hourly
        self.total_cong_ghg = self.total_car_cong_ghg + self.total_truck_cong_ghg
        self.total_car_freeflow_ghg = self.total_freeflow_hours * self.car_freeflow_ghg_hourly
        self.total_ghg = self.total_cong_ghg + self.total_car_freeflow_ghg
        self.total_ghg_sum = self.total_ghg_sum + self.total_ghg
    def transit_costs(self):
        self.train_cost_discount = min(2.0, 0.0001 * max(0,self.total_mode_counts["train"]  - self.train_baseline_ridership))
        self.train_cost_per_rider = self.train_cost_base - self.train_cost_discount
        self.bus_cost_discount = min(1.0, 0.00005 * max(0, self.total_mode_counts["bus"] - self.bus_baseline_ridership))
        self.bus_cost_per_rider = self.bus_cost_base - self.bus_cost_discount
        self.total_transit_cost = self.train_cost_per_rider * self.total_mode_counts["train"] + self.bus_cost_per_rider * self.total_mode_counts["bus"]
        self.train_rev = (1-self.fare_discount) * self.total_mode_counts["train"] * self.train_cost
        self.bus_rev = (1- self.fare_discount) * self.bus_cost * self.total_mode_counts["bus"]
        self.total_transit_rev = self.train_rev + self.bus_rev
        self.total_transit_profit = self.total_transit_rev - self.total_transit_cost
    def car_road_costs(self):
        self.road_maintainence = self.road_maintainence_cost_car  * (self.total_freeflow_hours + self.car_congestion_hours) + self.road_maintainence_cost_truck * self.truck_congestion_hours
        self.toll_revenue = self.total_mode_counts["car"] * self.car_toll
        self.toll_profit = self.toll_revenue * (1 - self.car_enforcement_pct) - self.road_maintainence
    #def calculate_shares()