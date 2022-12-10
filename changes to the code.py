###### Please read the comments carefully
####in file main.py：####
#at the beginning, use
from util import *

#add and complete the following function for Task 4.3:
def generate_traces_trained(env, function_name, agent, arrival_rate):

    #your policy in 1.2
    env.reset(function_name)
    state = env.reset_arrival_rate(function_name,arrival_rate)

    num_steps = 2000

    file = open("traces_your_policy.csv", "w")
    file.write('avg_cpu_util,slo_preservation,total_cpu_shares,cpu_shares_others,num_containers,arrival_rate,' +
               'vertical_scaling,horizontal_action,reward\n')

    total_reward = 0 
    for i in range(num_steps):
        
        # [Task 4.3] TODO:here use the same code you use for Task 1.2
        # [Your Code]


        #the flag is a random generator to determine which of vertical_action and horizontal_action to be zero
        flag = np.random.binomial(1, .5)
        if flag==1:
            vertical_action, horizontal_action = 0, np.random.randint(257)
        else:
            vertical_action, horizontal_action = np.random.randint(257), 0

        action = {'vertical':vertical_action,'horizontal':horizontal_action}
        next_state, reward, _ = env.step(function_name, action)



        total_reward += reward
        # print to file
        file.write(','.join([str(j) for j in state]) + ',' + str(vertical_action) + ',' + str(horizontal_action) +
                   ',' + str(reward) + '\n')
        state = next_state

    file.write('total_reward: ' + str(total_reward))

    print("your policy total reward: ", total_reward)
    file.close()


    #RL  policy  
    #env.reset(function_name)
    #state = env.reset_arrival_rate(function_name,arrival_rate)[:NUM_STATES]
    state = state[:NUM_STATES]

    file = open("traces_RL.csv", "w")
    file.write('avg_cpu_util,slo_preservation,total_cpu_shares,cpu_shares_others,num_containers,arrival_rate,' +
               'vertical_scaling,horizontal_action,reward\n')
    
    total_reward = 0 

    for i in range(num_steps):

        # [Task 4.3] TODO:here use the trained RL agent to generate the action for each step
        # hint: check calc_action in class PPO
        # [Your Code]


        total_reward += reward

        # print to file
        file.write(','.join([str(j) for j in state]) + ',' + str(action_to_execute['vertical']) + ',' + str(action_to_execute['horizontal']) +
                   ',' + str(reward) + '\n')
        state = next_state
        
    file.write('total_reward: ' + str(total_reward))

    print("RL policy total reward: ", total_reward)

    file.close()
    print('Trajectory generated!')


#at the end of main(), after agent.train(), add:
generate_traces_trained(env, function_name, agent, arrival_rate)
# where arrival_rate is a number between 1 and 8, and fixed after the initialization of the environment.
# For best performance of the RL agent, you should use the same arrival rate for training and generating the traces.
# you can use the following code to manually fix the arrival rate before the training, eg.
env.reset_arrival_rate(function_name,7)  # where 7 is the arrival rate that we arbitrarily decided





#### in file PPO.py #####
#replace the visualization function with the following for Task 4.1
def visualization(iteration_rewards, smoothed_rewards, smoothed_slo_preservations, smoothed_cpu_utils):
    # [Task 4.1] TODO: Write your code here to visualize the reward progression (learning curve), slo preservation and cpu utilization of the RL agent
    # [Task 4.1] TODO: Save the figure to a local file
    # [Your Code]
    pass