from mesa import Agent
import numpy as np

class CommuterAgent(Agent):

    def __init__(self, model, socio_group, income, car_owner, price_sensitivity, distance, time_value):
        super().__init__(model)
        self.socio_group = socio_group
        self.income = income
        self.car_owner = car_owner
        self.price_sensitivity = price_sensitivity
        self.mode_choice = None
        self.distance=distance
        self.time_value = time_value
        self.car_habit_streak = 0
        median_income = model.median_income  # e.g., 40000
        base_vot = 10.0  # $10/hour at median
        elasticity_vot = 1.0
        elasticity_cost = 1.0

        # Derived
        self.value_of_time = 0.1 * (base_vot) * (self.income / median_income) ** elasticity_vot  # $/min
        self.price_sensitivity = 0.1 * (median_income / self.income) ** elasticity_cost
        
        #transport mode utility constants
        self.base_commute_time = {
            "car": self.commute_time(self.distance, "car") * self.model.congestion_level,
            "bus": self.commute_time(self.distance, "bus") * self.model.congestion_level,
            "train": self.commute_time(self.distance, "train"),
            "bike_walk": self.commute_time(self.distance, "bike_walk")
        }

        self.base_commute_cost = {
            "car": model.car_cost + model.car_toll,
            "bus": model.bus_cost,
            "train": model.train_cost,
            "bike_walk": 0.0  # assumed free
        }
        mode_shares = {
            "car": 0.63,
            "bus": 0.17,
            "train": 0.12,
            "bike_walk": 0.8  # Reference mode
        }
        income_mode_shares = {
            "lower": {
                "car": 0.47,
                "bus": 0.25,
                "train": 0.10,
                "bike_walk": 0.18
            },
            "middle": {
                "car": 0.60,
                "bus": 0.15,
                "train": 0.15,
                "bike_walk": 0.10
            },
            "upper": {
                "car": 0.70,
                "bus": 0.05,
                "train": 0.20,
                "bike_walk": 0.05
            }
        }

        # Choose reference mode
        reference_mode = "bike_walk"

        # Compute utility constants (ASCs) relative to reference
        self.utility_constants = {}
        reference_mode = "bike_walk"
        shares = income_mode_shares[self.socio_group]

        for mode, share in shares.items():
            if mode == reference_mode:
                self.utility_constants[mode] = 0.0
            else:
                self.utility_constants[mode] = np.log(share / shares[reference_mode])

    def say_hi(self):
        print(f"Hi, I am an agent, you can call me {self.unique_id!s}.")

    def step(self):
        utilities = self.calculate_utilities()
        # Example mode choice logic
        new_mode = self.choose_mode(utilities)

        # Update model's mode counts
        previous_mode = self.mode_choice
        self.mode_choice = new_mode

        if previous_mode is not None:
            self.model.mode_counts[previous_mode][self.socio_group] -= 1
            self.model.total_mode_counts[previous_mode] -= 1
        self.model.mode_counts[new_mode][self.socio_group] += 1
        self.model.total_mode_counts[new_mode] += 1
        if self.mode_choice == "car":
            self.car_habit_streak += 1
        else:
            self.car_habit_streak = 0

    def calculate_utilities(self):

        utilities = {}

        base = 1.5
        bonus = base + 0.1 * self.car_habit_streak - 0.5 * (self.income / self.model.median_income)
        stickiness_bonus = max(0, bonus)

        if self.car_owner:
            curr_time = self.commute_time(self.distance, "car") * self.model.congestion_level
            time_delta = curr_time - self.base_commute_time["car"]

            curr_cost = self.model.car_cost + self.model.car_toll
            cost_delta = curr_cost - self.base_commute_cost["car"]

            utilities["car"] = self.utility_constants["car"] + stickiness_bonus \
            - self.price_sensitivity * cost_delta \
            - self.value_of_time * time_delta
        else:
            utilities["car"] = -np.inf

        # Bus utility
        curr_time = self.commute_time(self.distance, "bus") * self.model.congestion_level
        time_delta = curr_time - self.base_commute_time["bus"]

        curr_cost = self.model.bus_cost * (1.0 - self.model.fare_discount)
        cost_delta = curr_cost - self.base_commute_cost["bus"]

        utilities["bus"] = self.utility_constants["bus"] \
            - self.price_sensitivity * cost_delta \
            - self.value_of_time * time_delta

        # Train utility
        curr_time = self.commute_time(self.distance, "train")
        time_delta = curr_time - self.base_commute_time["train"]

        curr_cost = self.model.train_cost * (1.0 - self.model.fare_discount)
        cost_delta = curr_cost - self.base_commute_cost["train"]

        utilities["train"] = self.utility_constants["train"] \
            - self.price_sensitivity * cost_delta \
            - self.value_of_time * time_delta

        # Bike/Walk utility
        curr_time = self.commute_time(self.distance, "bike_walk")
        time_delta = curr_time - self.base_commute_time["bike_walk"]

        utilities["bike_walk"] = self.utility_constants["bike_walk"] \
            - self.value_of_time * time_delta
        return utilities
    
    def commute_time(self, distance, mode):
    # Minutes per mile (based on empirical averages)
        time_per_mile = {
            "car": 4.5,
            "bus": 4.8,
            "train": 5.6,
            "bike_walk": 15.0
        }
        return distance * time_per_mile[mode] / 60.0

    def choose_mode(self, utilities):
        # Normalize utilities for numerical stability (logit-safe)
        max_utility = max(utilities.values())
        adjusted_utilities = {m: utilities[m] - max_utility for m in utilities}

        # Compute exponentiated utilities
        exp_utilities = {m: np.exp(adjusted_utilities[m]) for m in utilities}
        total = sum(exp_utilities.values())

        # Compute choice probabilities
        probs = {m: exp_utilities[m] / total for m in utilities}

        # Draw mode based on probabilities
        modes = list(probs.keys())
        prob_values = list(probs.values())
        chosen_mode = np.random.choice(modes, p=prob_values)

        return chosen_mode
