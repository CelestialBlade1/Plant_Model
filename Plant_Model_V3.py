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

#Initialize q table and visit table
q_table=[[10 for i in range(n_action_choices)] for j in range(job_time)]
visit_table=[[0 for i in range(n_action_choices)] for j in range(job_time)]


def decide(J,i,epsilon): #Function that defines action taking
        
            
        if J//20 != 0:  
            epsilon=epsilon/(J//20)   #This makes sure exploration reduces at later episodes
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[i]) #exploitation
        else:
            action = np.random.randint(0, 2) #exploration
        return action 


for j in range(300): #loops through episodes
    
    
    t0=0
    CUMUL=0
    for i in range(1000):
        
        
        
        
        #AGENT PLAYS
        action=decide(j,i,EPSILON)
        
        
        
        
        #ENVIRONMENT PLAYS:
        
        prob_fail=0.05*(i-t0)
        fail_roll=random.uniform()
        
        if action==1: #maintainence triggers the probability to be reset
            
            t0=i+1
            reward=PROFIT-COST_M
        
        elif fail_roll < prob_fail and action==0 : #failure of plant causes loss,provided no maintainence was done
            reward=PROFIT-COST_R - 5*(i-t0)
            

        elif fail_roll >=prob_fail and action==0: #no failure, no maintainence
            reward=PROFIT - 5*(i-t0)
            
        
        


        #AGENT LEARNS:
        
        #GET NEW Q
        
        
        LEARNING_RATE=0.95
        current_q=q_table[i][action]  #gathers the arguments needed to apply q learning foormula
        if i<999:
            max_future_q=np.max(q_table[i+1])
        else:
            max_future_q=reward


    
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * (max_future_q)) #Q_LEARNING!!
        
        
        #UPDATE QTABLE
        q_table[i][action]=new_q
        visit_table[i][action]+=1


        #gather data for plot
        CUMUL+=reward
    
    if j%2==0:
        #print(CUMUL)   #prints total reward of run in 1 episode
        j_count.append(j)
        cumulrewards.append(CUMUL/EPISODE_TIME) #gathers avg reward of run to display in graph at end

plt.plot(j_count,cumulrewards)
plt.ylabel('Average Reward per time step') #This data is more or less approximate to what is given in the research paper. Generating that data is extremely time inefficient
plt.xlabel('Episode number')
plt.ylim(bottom=70,top=88)
plt.show()



        
