# Masters-Thesis
*Learning-Based Optimal Charging Discharging Strategy  for Electric Vehicles Under Vehicle-to-Grid Scheme*
## Abstract
Recent advances in electric vehicles (EV) technologies have raised the significance of vehicle- to-grid (V2G)
schemes in the smart grid domain, which allows bidirectional flows of energy and information between
consumers and suppliers. In the V2G scheme, each vehicle is viewed as a potential energy storage system 
(ESS) that can provide surplus energy to the grid. V2G is deemed especially useful in reducing the peak
demand and load shifting of utilities by working as a backup system for renewable energy. Thus, it is 
essential to intelligently manage charging and discharging according to electricity prices and users’
needs. On the one hand, users may take full advantage of prices when participating in the V2G demand 
response program. However, uncertainties such as user’s commuting behavior, charging preference, 
and energy needs make it challenging to determine the optimal charging strategy. This paper formulates 
the individual EV charging problem as a Markov DecisionProcess (MDP) without a defined transition
probability. A model-free reinforcement learning (RL) approach is proposed to learn the optimal
sequential charging decisions untiltheEVbattery reaches its end of life. The objective is to minimize
user’s charging cost and maximize the use of EV battery as the vehicle goes through various
charging/discharging cycles, while also taking into account the distances traveled by the vehicle. The
performance of the proposed algorithm is evaluated with real-world data, and the learned
charging/discharging strategy is examined to investigate the effectiveness of the proposed method.

* DDPG code: *ddpg_learn.py*, *replay_buffer.py*
* RL environment code (simulator): *train_env.py*
* Includes 3-page summary of the dissertation
