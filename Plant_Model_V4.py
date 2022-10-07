import numpy as np
from numpy import random
import time
import matplotlib.pyplot as plt

#defining constants as in book
DISCOUNT=0.1
EPSILON=0.8 
EPISODE_TIME=1000
PROB_CONST=0.05
COST_M=30
COST_R=120
PROFIT=100
n_action_choices=2
job_time=EPISODE_TIME
j_count=[]
cumulrewards=[]

#Initializing q_table and table that keeps count of how many times a state is visited
q_table=[[0 for i in range(n_action_choices)] for j in range(job_time)]
visit_table=[[0 for i in range(n_action_choices)] for j in range(job_time)]


def decide(j,i,epsilon):    #function of how agent decides between 0-nothing and 1-maintainence
        if j//40 != 0:
            epsilon=epsilon/(j//20)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[i])
        else:
            action = np.random.randint(0, 2)
        return action 


for j in range(300):
    c_curr=1 #defines the current value of the condition constant
    t0=0
    CUMUL=0
    for i in range(1000):
        
        
        
        
        #AGENT PLAYS
        action=decide(j,i,EPSILON)
        
        
        
        
        #ENVIRONMENT PLAYS:
        c_prev=c_curr
        prob_fail=0.05*(i-t0)
        fail_roll=random.uniform()
        
        if action==1:
            c_curr=1
            t0=i+1
            reward=PROFIT-COST_M
        
        elif fail_roll < prob_fail and action==0 :
            reward=PROFIT-COST_R - 5*(i-t0)
            c_curr= c_prev-0.1*random.uniform()

        elif fail_roll >=prob_fail and action==0:
            reward=PROFIT - 5*(i-t0)
            c_curr= c_prev-0.1*random.uniform()
        else:
            print("exception")
        
        


        #AGENT LEARNS:
        
        #GET NEW Q
        
        if (visit_table[i][action])//30 == 0:
             LEARNING_RATE=0.95
        else:
             LEARNING_RATE=0.9/((visit_table[i][action])//30)
        #LEARNING_RATE=0.95
        current_q=q_table[i][action]
        if i<999:
            max_future_q=np.max(q_table[i+1])
        else:
            max_future_q=reward
    
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * (max_future_q))
        
        
        #UPDATE QTABLE
        q_table[i][action]=new_q
        visit_table[i][action]+=1


        #gather data for plot
        CUMUL+=reward
    #if CUMUL<74000:
        #print(j)
    if j%2==0:
        print(CUMUL)   #prints total reward of run
        j_count.append(j)
        cumulrewards.append(CUMUL/EPISODE_TIME) #gathers avg reward of run to display in graph at end

plt.plot(j_count,cumulrewards)
plt.ylabel('Average Reward per time step') #This data is more or less approximate to what is given in the research paper. Generating that data is extremely time inefficient
plt.xlabel('Episode number')
plt.show()

