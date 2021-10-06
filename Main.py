from DDPG import DDPG
from gym.wrappers import Monitor
from tqdm import tqdm
from Env import BipedalWalker
from Surprise import Surprise

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import random
import gym

def plotRewards(rewards):

	rewards = np.array(rewards)

	plt.xlim((0, len(rewards)-1))
	plt.ylim((min(rewards),max(rewards)))
	plt.xlabel('Episodes')
	plt.ylabel('Rewards')

	x = [(i) for i in range(len(rewards))]
	y = [(rewards[j]) for j in range(len(rewards))]
	plt.plot(x, y, label='Rewards', color='chocolate')

def plotSteps(steps):

	rewards = np.array(steps)

	plt.xlim((0, len(steps)-1))
	plt.ylim((min(steps),max(steps)))
	plt.xlabel('Episodes')
	plt.ylabel('Steps')

	x = [(i) for i in range(len(steps))]
	y = [(rewards[j]) for j in range(len(steps))]
	plt.plot(x, y, label='Steps', color='chocolate')

Easter = Surprise()
init_var = 0.003
batch_size = 128
max_memory_size = 1000000
lrs = [0.001, 0.001]
taus = [0.003, 0.003]
gamma = 0.99
weight_decays = [1e-5,1e-5]
seed = 0
magic = False

Task = "Test"

if Task == "Train":

	env = BipedalWalker()
	env.seed(1)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	a_high = env.action_space.high[0]
	num_epochs = 3000   # >= 20
	max_steps = 1800
	agent = DDPG(state_dim, action_dim, a_high, lrs, taus, gamma, init_var,
				 weight_decays, batch_size, max_memory_size, seed)
	best_reward = -1000
	rewards = []
	steps = []
	max_number = 0
	train_good = 0
	rate_decline = False
	explor_rate = 0.8

	pbar = tqdm(range(num_epochs))
	for e in pbar:

		episode_rewards = 0
		episode_steps = 0
		observation = env.reset()

		if random.random() > explor_rate:
			exploration = False
		else:
			exploration = True

		for step in range(max_steps):

			env.render()
			state = np.float32(observation)
			action = agent.choose_action(state, exploration)
			new_observation, reward, done, info = env.step(action)
			episode_rewards += reward
			episode_steps += 1

			if not done:
				new_state = np.float32(new_observation)
				agent.replayBuffer.add(state, action, reward, new_state)
				observation = new_observation
				agent.learn()
			else:
				break

		rewards.append(episode_rewards)
		steps.append(episode_steps)

		if episode_rewards >= best_reward:
			best_reward = episode_rewards
			max_number = e
			torch.save(agent.actor.state_dict(), 'Actor.pth')
			torch.save(agent.critic.state_dict(), 'Critic.pth')
			torch.save(agent.target_actor.state_dict(), 'Actor_p.pth')
			torch.save(agent.target_critic.state_dict(), 'Critic_p.pth')
			print("")
			print("  Episode reward >= best_reward, model saved")
		else:
			print("")
			print("  Episode reward < best_reward, start next epoch")

		time.sleep(0.1)
		print("  Reward:" + str(episode_rewards) + " Steps:" + str(episode_steps) +" Best:" + str(best_reward))
		print('  Best reward in episode %d' %max_number)
		time.sleep(0.1)

		if episode_rewards >= 300:   # Early stop
			print("  Performance is good enough, stop training")
			break

		# if episode_rewards >= 260:
		# 	train_good += 1

		if rate_decline == True:
			if train_good == 5:
				explor_rate = 0.4
				print("")
				print("  Current exploration rate = 0.4")
			elif train_good == 10:
				explor_rate = 0.2
				print("")
				print("  Current exploration rate = 0.2")
			elif train_good == 15:
				explor_rate = 0.1
				print("")
				print("  Current exploration rate = 0.1")
			elif train_good == 20:
				explor_rate = 0.05
				print("")
				print("  Current exploration rate = 0.05")

	plotRewards(rewards)
	plt.legend()
	plt.grid()
	plt.show()

	plotSteps(steps)
	plt.legend()
	plt.grid()
	plt.show()

elif Task == "Test":

	env = BipedalWalker()
	# env = Monitor(env, './video', video_callable = lambda episode_id: True, force = True)
	env.seed(1)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	a_high = env.action_space.high[0]
	num_epochs = 40   # Should >= 4
	max_steps = 2000
	agent = DDPG(state_dim, action_dim, a_high, lrs, taus, gamma, init_var,
				 weight_decays, batch_size, max_memory_size, seed)
	best_reward = -1000
	rewards = []
	steps = []
	win_time = 0

	agent.actor.load_state_dict(torch.load('Actor.pth'))
	agent.critic.load_state_dict(torch.load('Critic.pth'))
	agent.target_actor.load_state_dict(torch.load('Actor_p.pth'))
	agent.target_critic.load_state_dict(torch.load('Critic_p.pth'))

	pbar = tqdm(range(num_epochs))
	for e in pbar:

		episode_rewards = 0
		episode_steps = 0
		observation = env.reset()
		exploration = False

		for step in range(max_steps):

			env.render()
			state = np.float32(observation)
			action = agent.choose_action(state, exploration)
			new_observation, reward, done, info = env.step(action)
			episode_rewards += reward
			episode_steps += 1

			if not done:
				new_state = np.float32(new_observation)
				agent.replayBuffer.add(state, action, reward, new_state)
				observation = new_observation
			else:
				break

		rewards.append(episode_rewards)
		steps.append(episode_steps)

		time.sleep(0.1)
		print("")
		print("  Reward:" + str(episode_rewards) + " Steps:" + str(episode_steps))
		time.sleep(0.1)

		if episode_rewards >= 260:
			win_time += 1

	time.sleep(0.1)

	if 0.5 * num_epochs > win_time >= 0.25 * num_epochs:
		print("  Oh, your training is effective")
	elif 0.75 * num_epochs > win_time >= 0.5 * num_epochs:
		print("  Amazing, the model is great")
	elif num_epochs > win_time >= 0.75 * num_epochs:
		print("  Perfect, this agent is wonderful")
	elif win_time == num_epochs and magic == False:
		print("  Full mark, unprecedented!")
	elif win_time == num_epochs and magic:
		print(Easter.Egg)

	plotRewards(rewards)
	plt.legend()
	plt.grid()
	plt.show()

	plotSteps(steps)
	plt.legend()
	plt.grid()
	plt.show()

else:

	print("Invalid input, no such mode")
