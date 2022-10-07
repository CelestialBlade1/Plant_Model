# Plant_Model
Repository for the Plant Model RL Simulation


## Drive Link
https://drive.google.com/drive/folders/1DuDjAQMcatmXeSnyktFslnuhQlBYB7wn?usp=sharing

## Github Link to Repo
https://github.com/CelestialBlade1/Plant_Model/

## Problem Description: 
An industrial plant degrades over time according to some function of time(varies with the stages). Determine the optimal policy which determines the optimal frequency of conducting maintainence on the plant given that the maintainence causes a negative reward.

### Action Space: 
To conduct maintainence or to do nothing

This problem is inspired by a research paper (link attached in repo) and is conducted in 4 stages. Each of the 4 stages are solved usinng a simple Q-Learning Algorithm without the use of any Neural network.

### Stage 1: 
The plant degrades with time. Maintainence incurs a small negative reward. Breakdown incurs a large negative reward. Regular operation grants positive reward per time step. Achieves research paper accuracy.

### Stage 2: 
This time we provide the system with a measure of its current condition. The failure probability function is now modified to involve the condition variable c. It is a simple one-step complexification of the previous problem.

### Stage 3: 
This is another one-step complexification of Stage 1- we simply make the reward decay after maintainence with time

### Stage 4: 
This is a combination of stage 2 and 3 together.

All of these stages are solved using Reinforcement Learning by Q-Learning. The comparison of the Obtained result and the research estimate is linked in the repo.
