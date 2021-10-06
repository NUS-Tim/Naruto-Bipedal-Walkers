import Utils
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):

	def __init__(self, num_states, num_actions, action_lim, init_var, seed):
		super(Actor, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.init_var = init_var
		self.num_states = num_states
		self.num_actions = num_actions
		self.action_lim = action_lim
		self.l1 = nn.Linear(num_states, 256)
		self.cs = nn.Conv1d(1, 1, 9, stride=2, padding=4)
		self.l2 = nn.Linear(128, 64)
		self.l3 = nn.Linear(64, 32)
		self.lout = nn.Linear(32, num_actions)

	def forward(self, state):
		if len(state.size()) == 1:
			state = state.unsqueeze(0)
		s = state.unsqueeze(1)
		x = F.relu(self.l1(s))
		x = F.relu(self.cs(x))
		x = F.relu(self.l2(x))
		x = F.relu(self.l3(x))
		action = torch.tanh(self.lout(x)).squeeze(0) * self.action_lim
		return action

class Critic(nn.Module):

	def __init__(self, num_states, num_actions, init_var, seed):
		super(Critic, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.init_var = init_var
		self.num_states = num_states
		self.num_actions = num_actions
		self.ls1 = nn.Linear(num_states, 512)
		self.cs = nn.Conv1d(1, 1, 9, stride=2, padding=4)
		self.ls2 = nn.Linear(256, 128)
		self.la = nn.Linear(num_actions, 128)
		self.lsa = nn.Linear(256, 128)
		self.lout = nn.Linear(128, 1)

	def forward(self, state, action):
		state = state.unsqueeze(1)
		if len(action.size()) == 2:
			action = action.unsqueeze(1)
		x = F.relu(self.ls1(state))
		x = F.relu(self.cs(x))
		x = F.relu(self.ls2(x))
		a = F.relu(self.la(action))
		z = torch.cat((x, a), dim=2)
		z = F.relu(self.lsa(z))
		z = self.lout(z)
		return z

class DDPG(object):

	def __init__(self, num_states, num_actions, action_lim, taus, lrs, gamma, init_var, weight_decays, batch_size,
				 max_memory_size, seed):
		self.num_states = num_states
		self.num_actions = num_actions
		self.action_lim = action_lim
		self.gamma = gamma
		self.taus = taus
		self.lrs = lrs
		self.batch_size = batch_size
		self.replayBuffer = Utils.MemoryBuffer(max_memory_size, self.batch_size)
		self.iter = 0
		self.noise = Utils.OrnsteinUhlenbeckProcess(self.num_actions, mu=0, theta=0.15, sigma=0.2)
		self.actor = Actor(self.num_states, self.num_actions, self.action_lim, init_var, seed+1)
		self.target_actor = Actor(self.num_states, self.num_actions, self.action_lim, init_var, seed+1)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lrs[0], weight_decay=weight_decays[0])
		self.critic = Critic(self.num_states, self.num_actions, init_var, seed)
		self.target_critic = Critic(self.num_states, self.num_actions, init_var, seed)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lrs[1], weight_decay=weight_decays[1])
		self.reinit()
		self.loss_function = F.smooth_l1_loss

	def choose_action(self, state, exploration=False):
		state = torch.from_numpy(state).float().unsqueeze(0)
		action = self.actor.forward(state)
		action = action.detach().data.numpy()[0]
		if exploration:
			action = action + (self.noise.sample() * self.action_lim)
		return action

	def learn(self):
		states, actions, rewards, next_states = self.replayBuffer.sample()
		self.Critic_optimizer(states, actions, rewards, next_states)
		self.Actor_optimizer(states)
		Utils.softModelUpdate(self.target_actor, self.actor, self.taus[0])
		Utils.softModelUpdate(self.target_critic, self.critic, self.taus[1])

	def Critic_optimizer(self, state, action, rewards, next_state):
		y_predicted = self.critic.forward(state, action)
		y_predicted = torch.squeeze(y_predicted)
		y_predicted = y_predicted.reshape(-1)
		next_action = self.target_actor(next_state)
		next_Q_value = torch.squeeze(self.target_critic.forward(next_state, next_action.detach()))
		y_expected = rewards + self.gamma * next_Q_value
		Critic_loss = self.loss_function(y_predicted, y_expected)
		self.critic_optimizer.zero_grad()
		Critic_loss.backward()
		self.critic_optimizer.step()

	def Actor_optimizer(self, state):
		loss_actor = - torch.sum(self.critic.forward(state, self.actor.forward(state)))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

	def reinit(self):
		Utils.copyModel(self.actor, self.target_actor)
		Utils.copyModel(self.critic, self.target_critic)
