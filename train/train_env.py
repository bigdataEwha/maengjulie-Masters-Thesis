
import gym
from ddpg_learn import DDPGagent
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import random
import numpy as np
from scipy.stats import truncnorm
import time
from datetime import timedelta

start_time = time.time()

class Env(gym.Env):
    STATE_ELEMENTS = 2
    STATES = ['price', 'energy']


    STATE_PRICE = 0
    STATE_ENERGY = 1

    ACTION_ELEMENTS = 1
    ACTION_CHARGE = 0

    SOH = 1

    ENERGY_MAX = 24

    anxiety_coefficient = 0.1
    hourly_degrade = -0.000203496439
    '''
    derivative of nonlinear degradation = -22143760514951022046028800/(5246127357238753253*x**(1502087/1000000)+377488202143273187543680*x+6790596808555307992614195200*x**(497913/1000000))
    average value of 1000 cycles: -0.00020349643957568318
    '''

    price_low = 0.005
    price_high = 0.023

    def __init__(self, price_t, none=None):
        super(Env, self).__init__()
        self.verbose = False
        self.viewer = none
        self.price_t = price_t

        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(Env.STATE_ELEMENTS,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-4,
            high=4,
            shape=(Env.ACTION_ELEMENTS,),
            dtype=np.float32
        ) #low 0 results in agent only picking 0

        #self.action_space

        self.seed(100)
        self.reset()

        self.state_log = []


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def draw_event(self):
        '''Drawing an event at random, with equal probabilities
        0=Driving event
        1=Station event
        '''
        event_list = [0, 1]
        d_prob = 70
        p_prob = 30
        draw = random.choices(event_list, weights=(d_prob, p_prob), k=1)[0]
        # draw = self.np_random.randint(0, 1)
        #draw = random.randint(0, 1)
        return draw


    def calc_cycle_degradation(self, action, energy):
        dod = 1-(energy/ self.ENERGY_MAX)
        if action == 0:
            cycle = 0
        else:
            charge_percent = round((abs(action) / self.ENERGY_MAX), 3)
            cycle = round(((charge_percent / 0.7) * dod), 2)
        return cycle

    def calc_nonlinear_degradation(self, cycle):
        ''' degradation curve fitted by nonlinear-4PL method; x=cycles'''
        if cycle == 0:
            deg = self.soh + self.hourly_degrade
        else:
            deg = -232.6692 + ((0.9986765 + 232.6692) / (1 + (cycle / 1186297000) ** 0.502087))  + self.hourly_degrade

        return deg



    def _eval_action(self, action, energy):
        chosen_act = action[Env.ACTION_CHARGE]
        if energy == 0:
            if chosen_act < 0:
                legal_act = -chosen_act
            else:
                legal_act = chosen_act
        elif energy == self.ENERGY_MAX:
            if chosen_act > 0:
                legal_act = -chosen_act
            else:
                legal_act = chosen_act
        elif energy + chosen_act < 0:
            legal_act = -energy
        elif energy + chosen_act > self.ENERGY_MAX:
            legal_act = self.ENERGY_MAX - energy
        else:
            legal_act = chosen_act

        return legal_act

    def distance_prob(self):
        distance_probabilities = [50]
        probs = []
        total = 50
        for i in range(5):
            val = np.random.randint(0, total)
            probs.append(val)
            total -= val
        probs.append(total)
        distance_probabilities.extend(probs)

        return distance_probabilities



    def step(self, action):
        self.last_action = action
        price = self.state[Env.STATE_PRICE]
        energy = self.state[Env.STATE_ENERGY] #0~24 kWh
        #drive_util = self.state[Env.STATE_DRIVE_UTIL]
        soh = self.soh
        driving_events = self.driving_events
        prices = self.prices
        distance_probs = self.prob_d



        # draw event
        draw = self.draw_event()

        if draw == 0:  # "driving" event
            distanceList = [0, 5, 10, 15, 20, 25, 30]

            while True:

                #required_distance = random.choices(distanceList, weights=(90, 1.5, 3, 2.5, 1.5, 1, 0.5))[0] # for testing
                required_distance = random.choices(distanceList, weights=tuple(i for i in distance_probs))[0]
                required_energy = required_distance / 5

                if required_energy == 0:
                    self.state[Env.STATE_ENERGY] = energy - required_energy
                    self.penalty = 0
                    self.soh += self.hourly_degrade
                    self.parked = True
                    self.drive_state = 1
                    prices.append(self.state[Env.STATE_PRICE])
                    driving_events.append((self.step_num, -required_energy))

                elif energy >= required_energy:
                    self.penalty = 0
                    self.state[Env.STATE_ENERGY] = energy - required_energy
                    self.cycle_count += self.calc_cycle_degradation(required_energy, energy)
                    self.soh = self.calc_nonlinear_degradation(self.cycle_count)
                    self.drive_state = 1
                    prices.append(self.state[Env.STATE_PRICE])
                    driving_events.append((self.step_num, -required_energy))
                else:
                    self.penalty = (required_energy - energy) * price

                if self.verbose:
                    print(f"Distance required: {required_distance}")
                    print(f"penalty: {self.penalty}")

                self.state[Env.STATE_PRICE] = round(self.np_random.uniform(low=self.price_low, high=self.price_high),
                                                    5)  # new price
                self.step_num += 1

                break



        '''if self.verbose:
            print(self.action_space)'''

        energy = self.state[Env.STATE_ENERGY]
        price = self.state[Env.STATE_PRICE]
        cycle = self.cycle_count

        #choose action and return legal action
        if self.drive_state ==1:
            legal_act = 0

        else:
            legal_act = self._eval_action(action, energy)

        if self.verbose:
            print(f"legal action: {legal_act}")


        # calculate cycle
        cycle_add = self.calc_cycle_degradation(legal_act, energy)
        cycle += cycle_add

        # calculated degradation
        if self.parked:
            soh = self.soh
        else:
            soh = self.calc_nonlinear_degradation(cycle)


        # calculate reward
        if self.soh < 0.8:
            reward = - 800/self.step_num #battery cost / cumulative time
        else:
            reward = -price * legal_act - self.penalty

        next_energy = energy + legal_act



        # update
        self.state[Env.STATE_PRICE] = round(self.np_random.uniform(low=self.price_low, high=self.price_high), 5)
        self.state[Env.STATE_ENERGY] = next_energy
        self.soh = soh
        self.cycle_count = cycle
        self.drive_state = 0
        self.parked = False

        # track progress
        if self.verbose:
            print("***End Step {}, \n State={}, Reward={}".format(self.step_num, self.state, reward))
            print(f"soh: {self.soh}")
        self.state_log.append(self.state + [self.soh])
        self.step_num += 1


        # termination
        done = self.soh < 0.8

        state = [x for x in self.state]
        #step_num = self.step_num
        #prices = self.prices

        return state, reward, done, {'legal action': [legal_act], 'price threshold': [self.price_t]}

    def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale = sd)

    def reset(self):
        self.soh = 1
        self.cycle_count = 0
        self.step_num = 0
        self.state = [0] * Env.STATE_ELEMENTS
        self.last_action = [0] * Env.ACTION_ELEMENTS
        self.state_log = []
        self.drive_state = 0
        self.penalty = 0
        self.parked = False

        price = round(self.np_random.uniform(low=self.price_low, high=self.price_high), 5)
        self.state[Env.STATE_PRICE] = price
        soc_reset = self.get_truncated_normal(mean=0.5, sd=0.1, low=0.5, upp=0.9).rvs()
        self.state[Env.STATE_ENERGY] = self.ENERGY_MAX * soc_reset # kWh
        #self.state[Env.STATE_ENERGY] = 24

        self.driving_events = []
        self.prices = []

        distance_probabilities = [50]

        self.prob_d = self.distance_prob()

        return np.array(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

'''Register environment class with TF-Agents'''
register(
    id='ev-experiment-v0',
    entry_point=f'{__name__}:Env',
)

env_name = 'ev-experiment-v0'


def main():

    max_episode_num = 1000  # 최대 에피소드 설정
    env = gym.make(env_name, price_t = None)
    agent = DDPGagent(env)  # DDPG 에이전트 객체

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()

    elapsed_time = time.time() - start_time
    print("Elapsed time: %s" % (str(timedelta(seconds=elapsed_time))))


if __name__=="__main__":
    main()