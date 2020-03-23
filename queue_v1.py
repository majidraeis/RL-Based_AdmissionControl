import gym
from gym import spaces
import random
import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import style
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

style.use("ggplot")
customer_color = '#191970'
line_color = '#00BFFF'

class QueueEnv(gym.Env):
    """
    Define a multi-server queue.
    The environment defines the admission control problem in a multi-server queue.
    Action: accept(1) or reject(0), State: 1-queue length 2-#of busy servers
    """

    def __init__(self, n_s, rho, QL_th):
        #         self.__version__ = "0.1.0"
        # General variables defining the environment
        self.n_servers = n_s
        self.rho = rho
        self.QL_th = QL_th
        self.n_jobs = 0
        self.ql_vec = [0]
        self.t_arr = 0  # First arrival time is 0 by default
        self.t_vec = [0.0]
        self.render_initiate = True
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(1000)
        self.empty_servers = np.arange(n_s)
        self.assigned_servers = []
        self.t_fin = []
        self.job_dict = {}
        self.job_dict[0] = {'Tw': 0.0}
        self.accepted_job_ind_vec = []
        self.waiting_vec = []
        self.job_index = 1
        self.cnt = 1
        self.MAX_STEPS = 10000
        self.last_entered_job = 0


    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        reward = self._get_reward(action)
        ob = self.n_jobs
        done = False if self.cnt < self.MAX_STEPS else True
        self.cnt += 1
        return ob, reward, done, {}

    def _take_action(self, action):
        # Queue length before taking the action (upon job arrival)
        self.ql = max(self.n_jobs - self.n_servers, 0)
        if action:
            if self.n_jobs < self.n_servers:
                t_ent = self.t_arr
                self.empty_servers = [x for x in range(self.n_servers) if x not in self.assigned_servers]
                self.assigned_servers = np.append(self.assigned_servers, random.choice(self.empty_servers))

            else:
                # finding the time that each server gets empty
                t_available = [np.max(self.t_fin[self.assigned_servers == i]) for i in range(self.n_servers)]
                # pick the earliest server available
                picked_server = np.argmin(t_available)
                t_ent = max(self.t_arr, t_available[picked_server])
                self.assigned_servers = np.append(self.assigned_servers, picked_server)

            t_s = self._service_gen()
            self.t_fin = np.append(self.t_fin, t_ent + t_s)
            self.n_jobs += 1
            self.job_dict[self.job_index] = {'Ta': self.t_arr, 'Td': t_ent + t_s, 'Ts': t_s, 'Tw': t_ent-self.t_arr,
                                             'Ba': self.ql}
            self.last_entered_job = self.job_index

        self.last_t_arr = self.t_arr
        self.t_arr += self._inter_arr_gen()
        served_jobs_ind = np.arange(len(self.t_fin))[np.array(self.t_fin) < self.t_arr]
        if len(np.array(env.t_fin) < env.t_arr):
            self.last_t_fins = self.t_fin[np.array(self.t_fin) < self.t_arr]
        else:
            self.last_t_fins = []
        self.n_jobs -= np.sum(np.array(self.t_fin) < self.t_arr)
        self.t_fin = np.delete(self.t_fin, served_jobs_ind)
        self.assigned_servers = np.delete(self.assigned_servers, served_jobs_ind)
        self.job_index += 1

    def _inter_arr_gen(self):
        lambda_a = self.n_servers * self.rho
        return np.random.exponential(1 / lambda_a)

    def _service_gen(self):
        lambda_s = 1.0
        return np.random.exponential(1 / lambda_s)

    def _get_reward(self, action):
        # Queue length after taking the action (right before the next arrival)
        self.ql = max(self.n_jobs - self.n_servers, 0)
        if action:
            if self.ql > self.QL_th:
                return -1000
            else:
                return 5
        else:
            return -5

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.ql_vec = [0]
        self.n_jobs = 0
        self.t_arr = 0  # First arrival time is 0 by default
        self.t_vec = [0]
        self.empty_servers = np.arange(self.n_servers)
        self.assigned_servers = []
        self.t_fin = []
        self.job_dict = {}
        self.job_index = 1
        self.accepted_job_ind_vec = []
        self.waiting_vec = []
        self.job_dict[0] = {'Tw': 0.0}
        self.last_entered_job = 0
        return self.n_jobs

    def _text(self):
        text = "QL = %d" % self.ql
        self.canvas.itemconfig(self.text, text=text)
        self.canvas.update

    def render(self, mode='human', close=False):  # rendering the queue after taking the action

        self._Queue_plot()
        self._QL_plot()
        self._waiting_plot()
        self._text()
        return

    def _Queue_plot(self):

        num_busy_servers = max(self.n_jobs - self.ql, 0)
        qHead = 300
        qTail = 150
        p2qDist = 2
        c2cDist = 4
        shift_up = 70
        cirR_o = 25  # Server circle outer radius
        cirR_i = 22  # Server circle inner radius
        cirC = [335, 200]
        linewidth = 2
        cirC[1] = cirC[1] - ((self.n_servers - 1) * cirR_o + (self.n_servers - 1) * c2cDist / 2)

        if self.render_initiate:
            self.root = tk.Tk()
            self.canvas = tk.Canvas(self.root, width=550, height=250)
            self.root.title('Queueing')
            self.canvas.pack()
            self.inner_circle = []
            self.canvas.create_line(qTail, 175-shift_up, qHead, 175-shift_up, width=linewidth, fill=line_color)
            self.canvas.create_line(qTail, 225-shift_up, qHead, 225-shift_up, width=linewidth, fill=line_color)
            self.canvas.create_line(qHead, 175-shift_up, qHead, 225-shift_up, width=linewidth, fill=line_color)
            for c in range(self.n_servers):
                self.canvas.create_oval(cirC[0] - cirR_o, cirC[1] - cirR_o-shift_up, cirC[0] + cirR_o,
                                        cirC[1] + cirR_o-shift_up, outline=line_color, width=linewidth)
                self.inner_circle.append(self.canvas.create_oval(cirC[0] - cirR_i, cirC[1] - cirR_i-shift_up, cirC[0]
                                         + cirR_i, cirC[1] + cirR_i-shift_up, outline='white',
                                         fill='white', width=linewidth))
                cirC[1] = cirC[1] + c2cDist + 2 * cirR_o
            self.queue_len = self.canvas.create_rectangle(qHead - p2qDist-p2qDist, 180-shift_up, qHead - p2qDist-p2qDist
                                                          , 220-shift_up, fill=customer_color)

            self.text = self.canvas.create_text((qTail+qHead)/2, 166-shift_up, fill="black", text="Queue length = 0")
        else:

            self.canvas.coords(self.queue_len, qHead - p2qDist-p2qDist - self.ql * 5, 180-shift_up,
                               qHead - p2qDist-p2qDist, 220-shift_up)

            for c in range(self.n_servers):
                if c+1 <= num_busy_servers:
                    self.canvas.itemconfig(self.inner_circle[c], fill=customer_color)
                    self.canvas.update()

                else:
                    self.canvas.itemconfig(self.inner_circle[c], fill='white', width=linewidth)
                    self.canvas.update()

                cirC[1] = cirC[1] + c2cDist + 2 * cirR_o

    def _QL_plot(self):
        # **************After taking the action*************
        # ***************** Queue Length *******************
        n_job_after_action = self.n_jobs + len(self.last_t_fins)
        ql_after_action = max(n_job_after_action - self.n_servers, 0)
        self.ql_vec = np.append(self.ql_vec, ql_after_action)
        self.t_vec = np.append(self.t_vec, self.last_t_arr)
        # ******** Departures until the next arrival **********
        ql_deps = n_job_after_action - np.arange(1, len(self.last_t_fins)+1)
        ql_deps = (ql_deps - self.n_servers) * (ql_deps - self.n_servers > 0)
        self.ql_vec = np.append(self.ql_vec, ql_deps)
        self.t_vec = np.append(self.t_vec, np.sort(self.last_t_fins))

        if self.render_initiate:

            self.figure1 = Figure(figsize=(5, 4))
            self.subplot1 = self.figure1.add_subplot(111)
            # Queue length after taking the action
            self.subplot1.step(self.t_vec, self.ql_vec, where='post', color='lightsteelblue')
            self.plot1 = FigureCanvasTkAgg(self.figure1, self.root)
            self.plot1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
            self.subplot1.set_xlabel('Time')
            self.subplot1.set_ylabel('Queue length')
        else:

            self.subplot1.clear()
            self.subplot1.step(self.t_vec, self.ql_vec, where='post', color='tab:blue')
            self.figure1.canvas.draw_idle()
            self.subplot1.set_xlabel('Time')
            self.subplot1.set_ylabel('Queue length')

    def _waiting_plot(self):
        # ***************** Waiting time ********************
        if self.render_initiate:

            self.accepted_job_ind_vec.append(self.last_entered_job)
            self.waiting_vec.append(self.job_dict[self.last_entered_job]['Tw'])
            self.figure2 = Figure(figsize=(5, 4))
            self.subplot2 = self.figure2.add_subplot(111)
            self.subplot2.plot(self.accepted_job_ind_vec, self.waiting_vec, color='tab:red')
            self.subplot2.set_xlabel('Job')
            self.subplot2.set_ylabel('Waiting time')
            self.plot2 = FigureCanvasTkAgg(self.figure2, env.root)
            self.plot2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
            self.render_initiate = False  # This must be run in the last plot function
        else:
            self.accepted_job_ind_vec.append(self.last_entered_job)
            self.waiting_vec.append(self.job_dict[self.last_entered_job]['Tw'])
            self.subplot2.clear()
            self.subplot2.plot(self.accepted_job_ind_vec, self.waiting_vec, color='tab:red')
            self.subplot2.set_xlabel('Job')
            self.subplot2.set_ylabel('Waiting time')
            self.figure2.canvas.draw_idle()


class Agent():
    def __init__(self, env):
        self.isCont = \
            type(env.action_space) == gym.spaces.box.Box
        if self.isCont:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape

        else:
            self.action_size = env.action_space.n
            print("Action size:", self.action_size)

    def get_action(self, state):
        if self.isCont:
            action = np.random.uniform(self.action_low, self.action_high, self.action_shape)
        else:
            action = random.choice(range(env.action_space.n))
        return action

class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        self.state_size = env.observation_space.n

        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.eps = 0.98
        self.build_model()

    def build_model(self):
        self.q_table = 1e-4 * np.random.random((self.state_size, self.action_size))

    def get_action(self, state):
        q_state = self.q_table[state]
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.eps else action_greedy

    def train(self, experience):
        state, action, next_state, reward, done = experience

        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discount_rate * np.max(q_next)

        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * q_update

        if done:
            self.eps *= 0.99


#******************  Demo  ********************
#**********************************************

render_flag = True
env = QueueEnv(3, 3, 10)
agent = QAgent(env)
state = env.reset()
done = False
while not done:
    action = agent.get_action(state)
    next_state, reward, done, info = env.step(action)
    agent.train((state, action, next_state, reward, done))
    state = next_state
    print("s:", state, "a:", action)
    if render_flag:
        env.render()

if render_flag:
    env.canvas.mainloop()
