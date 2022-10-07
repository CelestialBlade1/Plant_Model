import numpy as np
from numpy import random
import time
import matplotlib.pyplot as plt

#defining constants as in book
DISCOUNT=0.1
EPSILON=0.3  
EPISODE_TIME=1000
PROB_CONST=0.05
COST_M=30
COST_R=120
PROFIT=100
n_action_choices=2
job_time=EPISODE_TIME
j_count=[]
cumulrewards=[]

#Initiate lookup table for q values and number of times a situation was visited
q_table=[[0 for i in range(n_action_choices)] for j in range(job_time)]
visit_table=[[0 for i in range(n_action_choices)] for j in range(job_time)]


def decide(j,i,epsilon):  #this action is how the agent makes a decision.
        if j//200 != 0:
            epsilon=epsilon/(j//200)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[i])
        else:
            action = np.random.randint(0, 2)
        return action 


for j in range(3000):
    c_curr=1
    t0=0
    CUMUL=0
    for i in range(1000):
        
        
        
        
        #AGENT PLAYS
        action=decide(j,i,EPSILON)
        
        
        
        
        #ENVIRONMENT PLAYS:
        c_prev=c_curr
        prob_fail=max(0.02+0.05*(i-t0),1-c_curr) #incorporating formula as given
        fail_roll=random.uniform() #random number b/w 0 and 1 is generated
        
        if fail_roll < prob_fail and action==0 : #The plant fails and agent chooses to do nothing
            reward=PROFIT-COST_R
            c_curr= c_prev-0.1*random.uniform()

        elif fail_roll >=prob_fail and action==0: #The plant doesn't fail and the agent chooses to do nothing
            reward=PROFIT
            c_curr= c_prev-0.1*random.uniform()
        
        elif action==1: #The agent does a maintainence job
            c_curr=1
            t0=i+1
            reward=PROFIT-COST_M


        #AGENT LEARNS:
        
        #GET NEW Q
        
        if (visit_table[i][action])//300 == 0:
             LEARNING_RATE=0.95
        else:
             LEARNING_RATE=0.9/((visit_table[i][action])//100)
        #LEARNING_RATE=0.95
        current_q=q_table[i][action]
        max_future_q=np.max(q_table[i])
    
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * (max_future_q))
        
        
        #UPDATE QTABLE
        q_table[i][action]=new_q
        visit_table[i][action]+=1


        #gather data for plot
        CUMUL+=reward
    
    if j%100==0:
        #print(CUMUL)   #prints total reward of run
        j_count.append(j)
        cumulrewards.append(CUMUL/EPISODE_TIME) #gathers avg reward of run to display in graph at end

plt.plot(j_count,cumulrewards)
plt.ylabel("Cumulative Reward")
plt.show()



        
