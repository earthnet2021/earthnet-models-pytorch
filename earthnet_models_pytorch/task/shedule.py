
from typing import Optional, Union

import abc
import math

class WeightShedule:
    def __init__(self, initial_weight = 0, increase_from_step = 4000, weight_step = 1e-10, max_weight = 1, decreasing = False):
        self.weight = initial_weight
        self.increase_from_step = increase_from_step
        self.weight_step = weight_step
        self.max_weight = max_weight
        self.decreasing = decreasing

    def __call__(self, current_step = None):
        if current_step is not None:
            if current_step >= self.increase_from_step and ((self.weight < self.max_weight and not self.decreasing) or (self.weight > self.max_weight and self.decreasing)):
                self.weight += self.weight_step
        return self.weight

class CyclicShedule:
    def __init__(self, initial_value = 0, cycle_from_step = 1000, cycle_length = 1000, min_value = 0, max_value = 1, max_value_decrease = 0, n_cycles = 10): 
        self.value = initial_value
        self.min_value = min_value
        self.cycle_from_step = cycle_from_step
        self.cycle_length = cycle_length
        self.max_value = max_value
        self.max_value_decrease = max_value_decrease
        self.n_cycles = n_cycles

    def __call__(self, current_step = None):
        if current_step is not None:
            if (current_step >= self.cycle_from_step) and (((current_step-self.cycle_from_step)//self.cycle_length) < self.n_cycles):
                max_value = self.max_value * (1-self.max_value_decrease)**((current_step-self.cycle_from_step)//self.cycle_length)
                current_progress = ((current_step-self.cycle_from_step)%self.cycle_length)/self.cycle_length
                self.value = (max_value - self.min_value) * current_progress + self.min_value
            else:
                self.value = self.min_value
        return self.value

class ReverseSheduleSamplingExp:
    def __init__(self, r_sampling_step_1 = 25000, r_sampling_step_2 = 50000, r_exp_alpha = 2500):
        self.r_sampling_step_1 = r_sampling_step_1
        self.r_sampling_step_2 = r_sampling_step_2
        self.r_exp_alpha = r_exp_alpha

    def __call__(self, current_step = None):

        if current_step:
            if current_step < self.r_sampling_step_1:
                r_eta = 0.5
            elif current_step < self.r_sampling_step_2:
                r_eta = 1.0 - 0.5 * math.exp(-float(current_step - self.r_sampling_step_1) / self.r_exp_alpha)
            else:
                r_eta = 1.0

            if current_step < self.r_sampling_step_1:
                eta = 0.5
            elif current_step < self.r_sampling_step_2:
                eta = 0.5 - (0.5 / (self.r_sampling_step_2 - self.r_sampling_step_1)) * (current_step - self.r_sampling_step_1)
            else:
                eta = 0.0
        else:
            r_eta = 1.0
            eta = 0.0

        return r_eta, eta


SHEDULERS = {"cyclic": CyclicShedule, "linear": WeightShedule, "reverse_exp": ReverseSheduleSamplingExp}
