
import numpy as np
from numpy import random

import matplotlib.pyplot as plt


class Plant: ##This creates the plant. It doesn't  dispense the reward.
    def __init__(self, t0=0): 
        self.t0= t0                        #Equipped with a reference starting time when probability = 0. 
        self.failure_probability = 0.05    #Thereafter, P=0.05(t-t0)

    def failure_prob(self,P,t):            #Created the failure probability as an in-built attribute to plant
        return  P*(t-self.t0)

    def receive(self, timestamp, action):  #receives action - 0 is no maintainence, 1 is maintainence
        if action==0:
            return False
        elif action==1:
            self.t0=timestamp+1              #Clearly maintainence will bring t0 to the current time so that failure probability=0 again
            return True


class Agent:                                #The Agent, equipped with decision making, q_table and new_q writing attributes
    def __init__(self,n_action_choices,job_time):
       self.jobtime=job_time
       self.q_table=[[random.randint(-1,3) for i in range(n_action_choices)] for j in range(job_time)]
       self.visit_table=[[0 for i in range(n_action_choices)] for j in range(job_time)]
    
    def decide(self,timestamp,epsilon):
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(self.q_table[timestamp])
        else:
            action = np.random.randint(0, 2)
        return action 

    
    def update_q_table(self, action_choice, timestamp, new_q):
        self.q_table[timestamp][action_choice]=new_q
        self.visit_table[timestamp][action_choice]+=1

    
    def get_new_q(self, action_choice, timestamp, discount, reward):
        
        # if (self.visit_table[timestamp][action_choice])//300 == 0:
        #     LEARNING_RATE=0.95
        # else:
        #     LEARNING_RATE=0.9/((self.visit_table[timestamp][action_choice])//300)
        LEARNING_RATE=0.95
        current_q=self.q_table[timestamp][action_choice]
        max_future_q=np.max(self.q_table[timestamp])
    
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + discount * (max_future_q))

        return new_q

#defining constants as in book
DISCOUNT=0.1
EPSILON=0.9   
EPISODE_TIME=1000
PROB_CONST=0.05
COST_M=30
COST_R=120
PROFIT=100


A = Agent(2,EPISODE_TIME) #initialize agent
kl=[] ## two lists which store data for plotting pycharts
jcount=[]
for j in range(500):
 h = Plant(0)
 
 CUMUL=0
# working=True
 for i in range(EPISODE_TIME): 
    if(j//60!=0 and j%60==1):        ##newly added
      EPSILON=0.9/(j//30)
    action = A.decide(i,EPSILON)
    
    if h.receive(i,action)==1:
       reward=100-COST_M 
       #working =True

    elif (np.random.random() > h.failure_prob(PROB_CONST,i)): #rolls die to find if plant failed
        reward=PROFIT
        
    else :
        reward = PROFIT-COST_R  #if machine fails
       
    
    A.update_q_table(action,i,A.get_new_q(action,i,DISCOUNT,reward)) #updates q table in episode
    CUMUL+=reward
    #if j%100==0:
 if j%2==0:
    #print(CUMUL)   #prints total reward of run every 100 episodes
    jcount.append(j)
    kl.append(CUMUL/EPISODE_TIME) #gathers avg reward of run to display in graph at end


plt.plot(jcount,kl)
plt.ylabel('Avg Cumulative Reward per episode')
plt.ylim(bottom=72, top=88)
plt.show()
 
 

