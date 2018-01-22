# reinforcement learning utils
from __future__ import print_function
import numpy as np
import time
from collections import namedtuple
import sys

# global
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


# i/o
import logging
import os
from datetime import datetime

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1, credit, stock, value')
Memory_experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

logFileName = 'dqn.log'
if os.path.isfile(logFileName):
  os.rename(logFileName, logFileName+'_'+datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
# setting up the logfile
logger = logging.getLogger('max_credit_reward')
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler(logFileName)
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s , %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

logger.info('{0:>12},{1:>12},{2:>12},{3:>12},{4:>12},{5:>12},{6:>12}'.format('step', 'stock', 'credit', 'max_credit', 'sell', 'hold', 'buy'))

class Memory(object):

  def __init__(self, options_in):
    self.options = options_in
    self.capacity = self.options.capacity
    self.window_size = self.options.window_size
    self.memory_state0 = np.zeros((self.capacity, self.window_size, self.options.input_dim))
    self.memory_action = np.zeros(self.capacity)
    self.memory_reward = np.zeros(self.capacity)
    self.memory_terminal = np.zeros(self.capacity)
    self.memory_state1 = np.zeros((self.capacity, self.window_size, self.options.input_dim))
    self.memory_training = np.zeros(self.capacity)
    self.memory_credit = np.zeros(self.capacity)
    self.memory_value = np.zeros(self.capacity)
    self.position = 0


  def save(self, state0, recent_action, reward, terminal, state1, training, credit, value):
    """Saves a transition."""
    if len(self.memory) < self.capacity:
      self.memory_state0[self.position] = state0 #.flatten()
      self.memory_action[self.position] = recent_action
      self.memory_reward[self.position] = reward
      self.memory_terminal[self.position] = terminal
      self.memory_state1[self.position] = state1 #.flatten()
      self.memory_training[self.position] = training
      self.memory_credit[self.position] = credit
      self.memory_value[self.position] = value
    self.position = (self.position + 1) % self.capacity
    # print('self.position',self.position)

  def sample(self, batch_size):
    indexes = np.random.randint(self.memory_state0[0:self.position + 1].shape[0], size=batch_size)
    return Memory_experience(state0 = self.memory_state0[indexes, :],
                      action = self.memory_action[indexes],
                      reward = self.memory_reward[indexes],
                      state1 = self.memory_state1[indexes, :],
                      terminal1 = self.memory_terminal[indexes])
  
  def __len__(self):
    return len(self.memory)

  def reset_memory(self):
    self.capacity = self.capacity
    self.memory = []
    self.position = 0

class Env():  
  def __init__(self, X, Y, X_ref, Y_ref, credit, stock, ind):
      self.x = X
      self.y = Y
      self.x_ref = X_ref
      self.y_ref = Y_ref
      self.credit = credit
      self.start_credit = credit
      self.start_stock = stock
      self.stock = stock
      self.value = credit # Not correct, but for the time being its fine...
      self.ind = ind
      self.ind_start = ind
      self.state_shape = X.shape
      
      self._reset_experience()

  def _reset_experience(self):
    self.exp_state0 = None
    self.exp_action = None
    self.exp_reward = None
    self.exp_state1 = None
    self.exp_terminal1 = None
    self.exp_credit = self.start_credit
    self.exp_stock = self.start_stock
    self.exp_value = self.start_credit # Not correct, but for the time being its fine...


  def _get_experience(self):
    return Experience(state0 = self.exp_state0,
                      action = self.exp_action,
                      reward = self.exp_reward,
                      state1 = self.exp_state1,
                      terminal1 = self.exp_terminal1,
                      credit = self.exp_credit,
                      stock = self.exp_stock,
                      value = self.exp_value) 

  def reset(self):
    self._reset_experience()
    self.ind = self.ind_start
    self.exp_state1 = self.x[:, self.ind + 1, :]
    return self._get_experience()

  def eval_action(self, action, credit_diff, credit_max, value_max, value_min, credit_now, value_now):
    # diff =  self.y_ref[0, ind, 4] - self.x_ref[-1, ind, 4]
    # diff = self.x_ref[-1, ind, 4] - np.mean(self.y_ref[0:10, ind, 4])

    # if credit_diff < -2:
    #   r = 1
    # elif credit_diff > 2:
    #   r = -1
    # else:
    #   r = 0

    # Reward only if max credit is over hist. high
    # credit_max = credit_max * 1.01
    # if credit_now > credit_max:
    #   print('credit', credit_now)
    #   print('*****************************************')
    #   print('-------------------------------------------------------juhu>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    #   print('*****************************************')
    #   r = 1000
    # else:
    #   r = -1

    # Reward only if max value is over hist. high
    credit_max = credit_max * 1.01
    if value_now > value_max:
      print('value now', value_now)
      print('*****************************************')
      print('-------------------------------------------------------juhu>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
      print('*****************************************')
      r = value_now - value_max
    elif value_now < value_min:
      # print('value now', value_now)
      # print('*****************************************')
      # print('-------------------------------------------------------buuu>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
      r = value_now - value_min
    else:
      r = 0

#     if diff < 0. and action < 1:
#       r = -diff - 0.01 * self.x_ref[-1, ind, 4]
#     elif diff > 0. and action > 1 :
#       r = diff - 0.01 * self.x_ref[-1, ind, 4]
#     elif action == 1:
#       r = 0
#     else:
#       r = -10 * abs(diff)
    #print(value_now, value_max)

    return r

  def return_state(self):
      if self.exp_credit <= 0:
        self.exp_terminal1 = True
      if self.ind == (len(self.x_ref[1])-2):
        self.exp_terminal1 = True
        state = self.x[:, self.ind, :]
        new_state = self.x[:, self.ind + 1, :]
      else:
        self.exp_terminal1 = False
        state = self.x[:, self.ind, :]
        new_state = self.x[:, self.ind + 1, :]
      return state, new_state
  
  
  def do_step(self, action, credit_hist, value_hist):

    self.exp_action = action
    if action == 2 and self.exp_credit >= self.x_ref[-1, self.ind, 4]:
      self.exp_credit = self.exp_credit - self.x_ref[-1, self.ind, 4] - ( 0.0025 * self.x_ref[-1, self.ind, 4] )
      # print('self.x_ref[-1, self.ind, 4', self.x_ref[-1, self.ind, 4])
      self.exp_stock = self.exp_stock + 1

    elif action == 0 and self.exp_stock > 0 :
      self.exp_credit = self.exp_credit + self.x_ref[-1, self.ind, 4] - ( 0.0025 * self.x_ref[-1, self.ind, 4] )
      self.exp_stock = self.exp_stock - 1

    look_back = 120
    if self.ind > look_back:
      credit_diff = np.mean(credit_hist[self.ind-look_back:self.ind]) - credit_hist[self.ind]
    else:
      credit_diff = 1.

    credit_max = np.amax(credit_hist)
    # value_max = np.amax(value_hist)
    # value_min = np.amin(value_hist[0:self.ind+1])
    if self.ind >= 1000:
      value_max = np.amax(value_hist[self.ind-1000:self.ind])
      value_min = np.amin(value_hist[self.ind-1000:self.ind])
    else:
      if self.ind > 0:
        value_max = np.amax(value_hist[0:self.ind])
        value_min = np.amin(value_hist[0:self.ind])
      else:                                            # Etwas umsteandich...
        value_max = np.amax(value_hist)                 
        value_min = np.amin(value_hist)

    
    self.exp_value =  (self.exp_stock * self.x_ref[-1, self.ind, 4] ) + self.exp_credit

    self.exp_reward = self.eval_action(self.exp_action, credit_diff, credit_max, value_max, value_min, self.exp_credit, self.exp_value)
    self.exp_state0, self.exp_state1 = self.return_state()

    self.ind += 1
    return self._get_experience()


class DDQNAgent(object):

  def __init__(self, options_in, env_in, model_in, target_model_in, memory_in):

    self.options = options_in
    self.env = env_in
    self.state_shape = self.env.state_shape
    self.model = model_in
    self.target_model = target_model_in
    self.memory = memory_in
    self.optimiser = optim.Adam(self.model.parameters(), lr=self.options.lr) #, weight_decay=self.options.weight_decay)
    self.gamma = self.options.gamma
    self.loss = nn.SmoothL1Loss()
    self.target_model_update = 4

    self._reset_states()

  def _reset_states(self):
    self.memory.reset_memory()
    self.last_action = None
    self.last_observation = None

  def _update_target_model_hard(self):
    self.target_model.load_state_dict(self.model.state_dict())

  # # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
  # def _update_target_model_soft(self):
  #   for i, (key, target_weights) in enumerate(self.target_model.state_dict().iteritems()):
  #     target_weights += self.target_model_update * self.model.state_dict()[key]

  def _get_q_update(self, experiences): # compute temporal difference error for a batch
    # Start by extracting the necessary parameters (we use a vectorized implementation).
    # print(experiences.state0)
    state0_batch_vb = Variable(torch.cuda.FloatTensor(experiences.state0).permute(1,0,2)).cuda()
    action_batch_vb = Variable(torch.cuda.LongTensor(experiences.action.astype(int)))
    reward_batch_vb = Variable(torch.cuda.FloatTensor(experiences.reward))
    state1_batch_vb = Variable(torch.cuda.FloatTensor(experiences.state1).permute(1,0,2)).cuda()
    terminal1_batch_vb = 1 - experiences.terminal1.astype(int)
    terminal1_batch_vb = Variable(torch.cuda.FloatTensor(terminal1_batch_vb))
    # Compute target Q values for mini-batch update.
    # According to the paper "Deep Reinforcement Learning with Double Q-learning"
    # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
    # while the target network is used to estimate the Q value.
    q_values_vb = self.model(state1_batch_vb)
    # Detach this variable from the current graph since we don't want gradients to propagate
    _, q_max_actions_vb = q_values_vb.max(dim=1, keepdim=True)  # 0.2.0
    # Now, estimate Q values using the target network but select the values with the
    # highest Q value wrt to the online model (as computed above).
    next_max_q_values_vb = self.target_model(state1_batch_vb)
    # Detach this variable from the current graph since we don't want gradients to propagate
    next_max_q_values_vb = Variable(next_max_q_values_vb.data)
    next_max_q_values_vb = next_max_q_values_vb.gather(1, q_max_actions_vb)

    # Compute r_t + gamma * max_a Q(s_t+1, a) and update the targets accordingly
    # but only for the affected output units (as given by action_batch).
    current_q_values_vb = self.model(state0_batch_vb).gather(1, action_batch_vb.unsqueeze(1)).squeeze()
    # current_q_values_vb = Variable(current_q_values_vb.data)
    # current_q_values_vb = current_q_values_vb.gather(1, action_batch_vb.unsqueeze(1)).squeeze()
    # Set discounted reward to zero for all states that were terminal.
    next_max_q_values_vb = next_max_q_values_vb * terminal1_batch_vb.unsqueeze(1)
    expected_q_values_vb = reward_batch_vb + self.gamma * next_max_q_values_vb.squeeze()    # 0.2.0

    td_error_vb = self.loss(current_q_values_vb, expected_q_values_vb)

    # return v_avg, tderr_avg_vb
    if not self.training:   # then is being called from _compute_validation_stats, which is just doing inference
        td_error_vb = Variable(td_error_vb.data) # detach it from the graph
    return next_max_q_values_vb.data.clone().mean(), td_error_vb

  def _epsilon_greedy(self, q_values_ts):
    # calculate epsilon
    if self.training:  
      self.eps = self.eps_start - ((self.eps_start - self.eps_end) * self.eps_decay * self.step )
      if self.eps <= self.eps_end:
        self.eps = self.eps_end
      # if self.step % 1000 == 0:
      #   print('eps', self.eps)
    else:
      self.eps = self.eps_eval
    # choose action
    if np.random.uniform() < self.eps:  # then we choose a random action
      # BE CAREFUL the randint is hardcoded!!! maybe dumb in the future!?!?!?!?!?!?!
      action = np.random.randint(0, 3)
    else:                               # then we choose the greedy action
      if self.use_cuda:
        action = np.argmax(q_values_ts.cpu().numpy())
      else:
        action = np.argmax(q_values_ts.numpy())
    return action

  def _forward(self, state):

    if self.step < self.learn_start:  # then we don't do any learning, just accumulate experiences into replay memory
      # BE CAREFUL the randint is hardcoded!!! maybe dumb in the future!?!?!?!?!?!?!
      action = np.random.randint(0, 3)     # thus we only randomly sample actions here, since the model hasn't been updated at all till now
    else:
      state_ts = Variable(torch.cuda.FloatTensor(np.array(state)).unsqueeze(1), volatile=True)
      # print('state_ts.size(): ', state_ts.size())
      q_values_ts = self.model(state_ts).data # NOTE: only doing inference here, so volatile=True
      action = self._epsilon_greedy(q_values_ts)
    # Book keeping
    self.recent_state = state
    self.recent_action = action

    return action

  def _backward(self, reward, terminal):
    # Store most recent experience in memory.
    if self.step % self.memory_interval == 0:
      # NOTE: so the tuples stored in memory corresponds to:
      # NOTE: in recent_observation(state0), take recent_action(action), get reward(reward), ends up in terminal(terminal1)
      # memory.save works different here
      self.memory.save(self.recent_state, self.recent_action, reward, terminal, self.experience.state1, self.training, self.experience.credit, self.experience.value)

    if not self.training:
      return

    # Train the network on a single stochastic batch.
    if self.step > self.learn_start:
      exp = self.memory.sample(self.options.batch_size)
      # Compute temporal difference error
      _, td_error_vb = self._get_q_update(exp)
      self.optimiser.zero_grad()
      # run backward pass and clip gradient
      td_error_vb.backward()
      # for param in self.model.parameters():
      #     param.grad.data.clamp_(-self.clip_grad, self.clip_grad)
      # Perform the update
      self.optimiser.step()

    # # adjust learning rate if enabled
    # if self.lr_decay:
    #   self.lr_adjusted = max(self.lr * (self.steps - self.step) / self.steps, 1e-32)
    #   adjust_learning_rate(self.optimizer, self.lr_adjusted)

    if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
      self._update_target_model_hard()    # Hard update every `target_model_update` steps.

    return


  def fit(self):

    self.start_time = time.time()
    self.step = 0
    self.memory_interval = 1 # how often samples are stored in memory
    self.learn_start = (self.options.batch_size * self.memory_interval) + 5 #self.step + 10 # before this random action
    self.eps_decay = 0.001
    self.eps_start = 1
    self.eps_end = 0.05
    self.training = True
    self.use_cuda = True
    


    n_episodes = 0
    n_episodes_solved = 0
    reward = 0
    episode = 1
    episode_steps = None
    episode_reward = None
    total_reward = 0.
    start_over = True
    self.early_stop = False
    credit_hist = np.zeros(len(self.env.x_ref[1]))
    value_hist = np.zeros(len(self.env.x_ref[1]))
    action_hist = np.empty(1000)
    action_hist[:] = 4
    stock_hist = np.zeros(1000)
    counter = 0
    counter2 = 0

    while self.step < self.options.steps:
      if self.step%100 == 0 and self.step != 0:
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print('step: ', self.step)
        print('episode reward:', episode_reward)
        print('stock now:', self.experience.stock)
        # print('max stock:', np.amax(stock_hist))
        # print('min stock:', np.amin(stock_hist))
        print('credit now:', self.experience.credit)
        print('value now:', self.experience.value)
        # print('mean credit:', np.mean(credit_hist))
        print('max credit:', np.amax(credit_hist))
        if self.step > 1000:
          print('max value:', np.amax(value_hist[(self.step-1000):self.step]))
          print('min value:', np.amin(value_hist[(self.step-1000):self.step]))
        else:
          print('max value:', np.amax(value_hist[0:self.step]))
          print('min value:', np.amin(value_hist[0:self.step]))
        print('action count:', np.bincount(action_hist.astype(int))[0:3])
        print('---------------------------------------')
        # logger.info(str(self.step)+','+
        #             str(self.experience.stock)+','+
        #             str(self.experience.credit)+','+
        #             str(np.amax(credit_hist))+','+
        #             str(np.bincount(action_hist.astype(int))[0])+','+
        #             str(np.bincount(action_hist.astype(int))[1])+','+
        #             str(np.bincount(action_hist.astype(int))[2]))
        logger.info('{0:12d},{1:12d},{2:12.2f},{3:12.2f},{4:12d},{5:12d},{6:12d}'.format(self.step, self.experience.stock, self.experience.credit, np.amax(credit_hist), np.bincount(action_hist.astype(int))[0], np.bincount(action_hist.astype(int))[1], np.bincount(action_hist.astype(int))[2]) )
       
        stock_hist[:] = 0
        action_hist[:] = 4
        counter = 0
      # print('step', self.step)

      if start_over:    # start of a new episode
        print('-------------NEW EPISODE-------------')
        episode_steps = 0
        episode_reward = 0.
        episode += 1
        # Obtain the initial observation by resetting the environment
        self._reset_states()
        self.experience = self.env.reset()
        start_over = False
        credit_hist[:] = 0
        value_hist[:] = 0
        counter2 = 0
      # Run a single step
      # This is where all of the work happens. We first perceive and compute the action
      # (forward step) and then use the reward to improve (backward step)

      action = self._forward(self.experience.state1)
      action_hist[counter] = action
      stock_hist[counter] = self.experience.stock
      reward = 0.

      credit_hist[counter2] = self.experience.credit
      value_hist[counter2] = self.experience.value

      self.experience = self.env.do_step(action, credit_hist, value_hist)

      reward = self.experience.reward
      # print('self.experience.terminal1',self.experience.terminal1)
      if self.experience.terminal1:
        start_over = True

      self._backward(reward, self.experience.terminal1)

      episode_steps += 1
      episode_reward += reward
      self.step += 1
      counter += 1
      counter2 += 1
      # if self.step % 1000 == 0:


    self.end_time = time.time()  
    print('training time', self.end_time-self.start_time, 's')
