import random
from collections import defaultdict, namedtuple, deque
from itertools import product, starmap

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from matplotlib import animation
import copy
from numpy import sin, cos, pi


class CartPole:
    def __init__(self):
        self.grav = 9.81
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.mass_total = self.mass_cart + self.mass_pole
        self.pole_mcenter = 0.5
        self.polemass_mom = self.mass_pole * self.pole_mcenter
        self.force_mag = 10.0
        self.delta_t = 0.02
        self.lim_theta = pi / 15
        self.start_state = self.compute_start_state()
        self.range_theta_rad = 12 * 2 * pi / 360
        self.lim_x = 2.4
        self.n_actions = 2
        self.actions = [-1, 1]

    @property
    def state(self):
        return self._state

    def is_terminal(self, state):
        x, theta = state[:2]
        if (abs(x) >= self.lim_x) or (abs(theta) >= self.lim_theta):
            return True
        else:
            return False

    def reward(self, state):
        if self.is_terminal(state) is True:
            return 0
        else:
            return 1

    def newstate(self, state, action):
        move = self.actions[action]
        x, theta, x_dot, theta_dot = state
        F = self.force_mag * move
        costheta = cos(theta)
        sintheta = sin(theta)
        temp = (F + self.polemass_mom * theta_dot * theta_dot * sintheta
                ) / self.mass_total

        theta_dot_dot = (self.grav * sintheta - costheta * temp) / (
            self.pole_mcenter *
            (4.0 / 3.0 - self.mass_pole * costheta * costheta / self.mass_total
             ))

        x_dot_dot = temp - self.polemass_mom * theta_dot_dot * costheta / self.mass_total
        x_new = x + self.delta_t * x_dot
        x_dot_new = x_dot + self.delta_t * x_dot_dot
        theta_new = theta + self.delta_t * theta_dot
        theta_dot_new = theta_dot + self.delta_t * theta_dot_dot
        return (x_new, theta_new, x_dot_new, theta_dot_new)

    def compute_start_state(self):
        random = np.random.uniform(low=-0.05, high=0.05, size=(4, ))
        start_state = State(*random)
        return start_state
