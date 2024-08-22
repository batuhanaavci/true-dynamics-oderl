import torch
import numpy as np
import matplotlib.pyplot as plt
import envs
from policies import Policy
import time
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchdiffeq import odeint
from basic_mdl import basic_mdl
from utils import *
import os
import random
import copy



device = 'cuda' if torch.cuda.is_available() else 'cpu'
dt      = 0.1 		# mean time difference between observations
noise   = 0.0	# observation noise std
ts_grid = 'fixed' 	# the distribution for the observation time differences: ['fixed','uniform','exp']
ENV_CLS = envs.CTCartpole # [CTPendulum, CTCartpole, CTAcrobot]
env = ENV_CLS(dt=dt, obs_trans=False, device=device, obs_noise=noise, ts_grid=ts_grid,solver="rk4")

policy_nn = Policy(env)
policy_nn.reset_parameters()
policy_nn.to(device)

us_V = basic_mdl(env.n, 1, n_hid_layers=2, act="tanh", n_hidden=200)
us_V.reset_parameters()
us_V.to(device)

policy_opt = optim.Adam(policy_nn.parameters(), lr=0.001)
us_V_opt = optim.Adam(us_V.parameters(), lr=0.001)

# Training parameters
num_episodes =250
num_steps = 20
num_rounds = 50
gamma = 0.99
tau = 5.0

experience_buffer = []


for rounds in range(num_rounds):

    rewards, opt_objs = [], []
    for episode in range(num_episodes):
        if episode % 50 == 0:
            Vtarget = copy.deepcopy(us_V)

        # Draw initial states from previous experiences
        import random

        # Draw initial states from previous experiences
        if len(experience_buffer) >= 50:
            initial_observations = random.sample(experience_buffer, 50)

        else:
            # If buffer is empty or not enough experiences, reset the environment
            initial_observations = [env.reset() for _ in range(50)]
    
        
        s0 = torch.stack([env.obs2state(torch.tensor(obs, device=device)) for obs in initial_observations])
        ts = env.build_time_grid(num_steps).to(device)
        policy_opt.zero_grad()

        st, at, rt, ts = env.integrate_system(T=num_steps, g=policy_nn, s0=s0, N=1)
        rew_int = rt[:, -1].mean(0)  # N

        st_contiguous = st.contiguous()

        # Reshape the tensor to combine the first two dimensions
        st_flattened = st_contiguous.reshape(-1, st.size(-1))  # This results in shape (1000, 4)

        # Randomly sample 50 states from the flattened tensor
        indices = torch.randperm(st_flattened.size(0))[:50]
        random_states = st_flattened[indices]
        experience_buffer.extend(random_states.detach().cpu().numpy())

        # Expand st if only one trajectory is present
        st = torch.cat([st] * 5) if st.shape[0] == 1 else st
        ts = ts[0]
        gammas = (-ts / tau).exp()  # H
        V_st_gam = us_V(st.contiguous())[:, 1:, 0] * gammas[1:]  # L,N,H-1
        V_const = min(rounds / 5.0, 1)
        n_step_returns = rt[:, 1:] + V_const * V_st_gam  # ---> n_step_returns[:,:,k] is the sum in (5)
        optimized_returns = n_step_returns.mean(-1)  # L,N
        mean_cost = -optimized_returns.mean()
        mean_cost.backward()
        grad_norm = torch.norm(flatten_([p.grad for p in policy_nn.parameters()])).item()
        policy_opt.step()

        rewards.append(rew_int.mean().item())
        opt_objs.append(mean_cost.mean().item())
        print_log = 'Round: {:4d}/{:<4d}, Iter:{:4d}/{:<4d},  opt. target:{:.3f}  mean reward:{:.3f}  '\
                    .format(rounds, num_rounds, episode, num_episodes, np.mean(opt_objs), np.mean(rewards)) + \
                    'H={:.2f},  grad_norm={:.3f},  '.format(2.0, grad_norm)

        with torch.no_grad():
            # Regress all intermediate values
            last_states = st.detach().contiguous()[:, 1:, :]  # L,N,T-1,n
            last_values = Vtarget(last_states).squeeze(-1)
            Vtargets = rt[:, 1:].squeeze() + (-ts[1:] / tau).exp() * last_values  # L,N,T-1
            Vtargets = Vtargets.mean(-1)

        mean_val_err = 0
        for inner_iter in range(10):
            us_V_opt.zero_grad()
            td_error = us_V(s0).squeeze(-1) - Vtargets  # L,N
            td_error = torch.mean(td_error**2)
            td_error.backward()
            mean_val_err += td_error.item() / 10
            us_V_opt.step()

        if episode % (num_episodes // 5) == 0:
            print(print_log)
            if not os.path.exists('models/'):
                os.makedirs('models/')
            if not os.path.exists('models/policy9'):
                os.makedirs('models/policy9')
            if not os.path.exists('models/value9'):
                os.makedirs('models/value9')

            print("Saving model at round {} and episode {}".format(rounds, episode))
            torch.save(policy_nn.state_dict(), 'models/policy9/round_{}episode{}.pt'.format(rounds, episode))
            torch.save(us_V.state_dict(), 'models/value9/round_{}episode{}.pt'.format(rounds, episode))

    with torch.no_grad():
        Htest, Ntest, Tup = 30, 10, int(3.0 / 0.1)
        initial_observations = [env.reset() for _ in range(10)]
        s0 = torch.stack([env.obs2state(torch.tensor(obs, device=device)) for obs in initial_observations])
        test_states, test_actions, test_rewards, _ = env.integrate_system(T=200, s0=s0, g=policy_nn)
        
        true_test_rewards = test_rewards[..., Tup:].mean().item()
        print("True test rewards: ", true_test_rewards)
        
        if true_test_rewards > 0.9:
            print("Test rewards > 0.9. Training complete...")
            break

    env.close()