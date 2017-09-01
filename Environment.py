import numpy as np
import time
import sys
from random import randint
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
ENV_H = 10  # grid height
ENV_W = 10 # grid width


class Environment(tk.Tk, object):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r','s']
        self.n_actions = len(self.action_space)
        self.title('Environment')
        self.geometry('{0}x{1}'.format(ENV_H * UNIT, ENV_H * UNIT))
        self._build_maze()
        self.step_counter = 0

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='blue',
                           height=ENV_H * UNIT,
                           width=ENV_W * UNIT)

        # create grids
        for c in range(0, ENV_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, ENV_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, ENV_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, ENV_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([UNIT/2, UNIT/2])

        # hell
        hell1_center = origin + np.array([UNIT , UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - (UNIT * 3 / 8), hell1_center[1] - (UNIT * 3 / 8),
            hell1_center[0] + (UNIT * 3 / 8), hell1_center[1] + (UNIT * 3 / 8),
            fill='red')

        # create food oval
        oval_center = origin + UNIT * 5
        self.oval = self.canvas.create_oval(
            oval_center[0] - (UNIT * 3 / 8), oval_center[1] - (UNIT * 3 / 8),
            oval_center[0] + (UNIT * 3 / 8), oval_center[1] + (UNIT * 3 / 8),
            fill='yellow')

        # create agent rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - (UNIT * 3 / 8), origin[1] - (UNIT * 3 / 8),
            origin[0] + (UNIT * 3 / 8), origin[1] + (UNIT * 3 / 8),
            fill='black')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([UNIT/2, UNIT/2])
        self.rect = self.canvas.create_rectangle(
            origin[0] - (UNIT * 3 / 8), origin[1] - (UNIT * 3 / 8),
            origin[0] + (UNIT * 3 / 8), origin[1] + (UNIT * 3 / 8),
            fill='black')

        self.move_food()
        self.move_hell()
        while self.canvas.coords(self.oval) == self.canvas.coords(self.hell1):
            self.move_food()
            self.move_hell()

        # return observation
        return np.array(self.canvas.coords(self.rect)[0:2] + self.canvas.coords(self.oval)[0:2] + self.canvas.coords(self.hell1)[0:2])

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (ENV_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (ENV_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state
        state = self.canvas.coords(self.rect)[0:2] + self.canvas.coords(self.oval)[0:2] + self.canvas.coords(self.hell1)[0:2]

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 100
            done = False
        elif s_ == self.canvas.coords(self.hell1):
            reward = -20
            done = False
        else:
            reward = -1
            done = False

        if self.step_counter == 25:
            done = True
            self.step_counter = 0

        self.step_counter += 1

        return np.array(state), reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

    def move_food(self):
        self.canvas.delete(self.oval)
        origin = np.array([UNIT/2, UNIT/2])
        oval_center = origin + np.array([UNIT * randint(1, ENV_W - 2), UNIT * randint(1, ENV_H - 2)])
        self.oval = self.canvas.create_oval(
                oval_center[0] - (UNIT * 3 / 8), oval_center[1] - (UNIT * 3 / 8),
                oval_center[0] + (UNIT * 3 / 8), oval_center[1] + (UNIT * 3 / 8),
                fill='yellow')

    def move_hell(self):
        self.canvas.delete(self.hell1)
        origin = np.array([UNIT/2, UNIT/2])
        hell1_center = origin + np.array([UNIT * randint(1, ENV_W - 2), UNIT * randint(1, ENV_H - 2)])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - (UNIT * 3 / 8), hell1_center[1] - (UNIT * 3 / 8),
            hell1_center[0] + (UNIT * 3 / 8), hell1_center[1] + (UNIT * 3 / 8),
            fill='red')

def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            #print(s,"\n",r,"\n",done,"\n")
            if done:
                break

if __name__ == '__main__':
    env = Environment()
    env.after(100, update)
    env.mainloop()