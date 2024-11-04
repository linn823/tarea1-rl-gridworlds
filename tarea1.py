import matplotlib.pyplot as plt
import numpy as np
import gym
import gym_gridworld
import time

env = gym.make("GridWorld-v0")
env.verbose = True
_ =env.reset()

EPISODES = 10000 
MAX_STEPS = 150

LEARNING_RATE = 0.01
GAMMA = 1
#LEARNING_RATE = 0.2  
#GAMMA = 0.90 

LAMBDA = 0.5

epsilon = 1

print('Observation space\n')
print(env.observation_space)


print('Action space\n')
print(env.action_space)

rewards = []

def qlearning(env, epsilon):
    STATES = env.n_states  # Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions  # Cantidad de acciones posibles.

    Q = np.zeros((STATES, ACTIONS))  # Inicializa la Q table con 0s.
    for episode in range(EPISODES):
        rewards_epi = 0
        state = env.reset()  # Reinicia el ambiente
        for actual_step in range(MAX_STEPS):

            # Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                # De lo contrario, escogerá el estado con el mayor valor.
                action = np.argmax(Q[state, :])

            # Ejecuta la acción en el ambiente y guarda los nuevos parámetros (estado siguiente, recompensa, ¿terminó?).
            next_state, reward, done, _ = env.step(action)
            rewards_epi = rewards_epi+reward

            # Calcula la nueva Q table.
            Q[state, action] = Q[state, action] + LEARNING_RATE * \
                (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state

            if (MAX_STEPS-2) < actual_step:
                #print(f"Episode {episode} rewards: {rewards_epi}")
                
                # Guarda las recompensas en una lista
                rewards.append(rewards_epi)
                #print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001

            if done:
                #print(f"Episode {episode} rewards: {rewards_epi}")
                
                # Guarda las recompensas en una lista
                rewards.append(rewards_epi)
                #print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001
                break

    #print(Q)
    # print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    return Q

def dQlearning(env, epsilon):
    STATES = env.n_states  # Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions  # Cantidad de acciones posibles.

    Q1 = np.zeros((STATES, ACTIONS))  # Inicializa la Q1 table con 0s.
    Q2 = np.zeros((STATES, ACTIONS))  # Inicializa la Q2 table con 0s.

    for episode in range(EPISODES):
        rewards_epi = 0
        state = env.reset()  # Reinicia el ambiente
        for actual_step in range(MAX_STEPS):

            p = np.random.random()
            if p < 0.5:
                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q1[state, :])

                next_state, reward, done, _ = env.step(action)
                rewards_epi = rewards_epi+reward

                # Calcula la nueva Q1 table.
                Q1[state, action] = Q1[state, action] + LEARNING_RATE * \
                    (reward + GAMMA * np.max(Q2[next_state, :]) - Q1[state, action])

                state = next_state
            else:
                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q2[state, :])

                next_state, reward, done, _ = env.step(action)
                rewards_epi = rewards_epi+reward

                # Calcula la nueva Q2 table.
                Q2[state, action] = Q2[state, action] + LEARNING_RATE * \
                    (reward + GAMMA * np.max(Q1[next_state, :]) - Q2[state, action])

                state = next_state

            if (MAX_STEPS-2) < actual_step:
                #print(f"Episode {episode} rewards: {rewards_epi}")
                
                # Guarda las recompensas en una lista
                rewards.append(rewards_epi)
                #print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001

            if done:
                #print(f"Episode {episode} rewards: {rewards_epi}")
                
                # Guarda las recompensas en una lista
                rewards.append(rewards_epi)
                #print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001
                break

    return Q1


def lambdaQlearning(env, epsilon):
    STATES = env.n_states  # Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions  # Cantidad de acciones posibles.

    Q = np.zeros((STATES, ACTIONS))  # Inicializa la Q table con 0s.
    E = np.zeros((STATES, ACTIONS))

    for episode in range(EPISODES):
        
        rewards_epi = 0
        state = env.reset()  # Reinicia el ambiente

        ## primera accion
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        for actual_step in range(MAX_STEPS):
            
            next_state, reward, done, _ = env.step(action)
            rewards_epi = rewards_epi+reward
            
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state, :])

            next_state_best_action = np.argmax(Q[next_state, :])
            delta = reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action]
            E[state, action] = E[state, action] + 1

            # Calcula la nueva Q table y E table.
            for e_state in range(STATES):
                for e_action in range(ACTIONS):
                    Q[e_state, e_action] = Q[e_state, e_action] + LEARNING_RATE * delta * E[e_state, e_action]

                    if next_action == next_state_best_action:
                        E[e_state,e_action] = GAMMA * LAMBDA * E[e_state, e_action]
                    else:
                        E[e_state,e_action] = 0

            state = next_state
            action = next_action

            if (MAX_STEPS-2) < actual_step:
                #print(f"Episode {episode} rewards: {rewards_epi}")
                
                # Guarda las recompensas en una lista
                rewards.append(rewards_epi)
                #print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001

            if done:
                #print(f"Episode {episode} rewards: {rewards_epi}")
                
                # Guarda las recompensas en una lista
                rewards.append(rewards_epi)
                #print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001
                break

    #print(Q)
    # print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    return Q

def sarsa(env, epsilon):
    STATES = env.n_states  # Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions  # Cantidad de acciones posibles

    Q = np.zeros((STATES, ACTIONS))

    for episode in range(EPISODES):
        rewards_epi = 0
        state = env.reset()  # Reinicia el ambiente

        # Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # De lo contrario, escogerá el estado con el mayor valor.
            action = np.argmax(Q[state, :])

        for actual_step in range(MAX_STEPS):

            next_state, reward, done, _ = env.step(action)

            # Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
            if np.random.uniform(0, 1) < epsilon:
                action2 = env.action_space.sample()
            else:
                # De lo contrario, escogerá el estado con el mayor valor.
                action2 = np.argmax(Q[next_state, :])

            # Calcula la nueva Q table.
            Q[state, action] = Q[state, action] + LEARNING_RATE * \
                (reward + GAMMA * Q[next_state, action2] - Q[state, action])
            rewards_epi = rewards_epi+reward
            state = next_state
            action = action2

            if (MAX_STEPS-2) < actual_step:
                #print(f"Episode {episode} rewards: {rewards_epi}")
                rewards.append(rewards_epi)
                
                #print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001

            if done:
                #print(f"Episode {episode} rewards: {rewards_epi}")
                # Guarda las recompensas en una lista
                rewards.append(rewards_epi)
                #print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001
                break

    #print(Q)
    # print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    return Q

def lambdaSarsa(env, epsilon):
    STATES = env.n_states  # Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions  # Cantidad de acciones posibles

    Q = np.zeros((STATES, ACTIONS))
    E = np.zeros((STATES, ACTIONS))

    for episode in range(EPISODES):
        rewards_epi = 0
        state = env.reset()  # Reinicia el ambiente

        ## primera accion
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        for actual_step in range(MAX_STEPS):

            next_state, reward, done, _ = env.step(action)

            # Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, escoge una acción al azar.
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                # De lo contrario, escogerá el estado con el mayor valor.
                next_action = np.argmax(Q[next_state, :])

            # Calcula la nueva Q table y E table.
            delta = (reward + GAMMA * Q[next_state, next_action] - Q[state, action])
            E[state, action] = E[state, action] + 1

            for e_state in range(STATES):
                for e_action in range(ACTIONS):
                    Q[e_state, e_action] = Q[e_state, e_action] + LEARNING_RATE * delta * E[e_state, e_action]
                    E[e_state,e_action] = GAMMA * LAMBDA * E[e_state, e_action]


            rewards_epi = rewards_epi+reward
            state = next_state
            action = next_action

            if (MAX_STEPS-2) < actual_step:
                #print(f"Episode {episode} rewards: {rewards_epi}")
                rewards.append(rewards_epi)
                
                #print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001

            if done:
                #print(f"Episode {episode} rewards: {rewards_epi}")
                # Guarda las recompensas en una lista
                rewards.append(rewards_epi)
                #print(f"Value of epsilon: {epsilon}")
                if epsilon > 0.1:
                    epsilon -= 0.0001
                break

    #print(Q)
    # print(f"Average reward: {sum(rewards)/len(rewards)}:") #Imprime la recompensa promedio.
    return Q

#Función para correr juegos siguiendo una determinada política
def playgames(env, Q, num_games, render = True):
    wins = 0
    env.reset()
    time.sleep(0.0016)
    env.render()

    for i_episode in range(num_games):
        rewards_epi=0
        observation = env.reset()
        t = 0
        while True:
            action = np.argmax(Q[observation, :]) #La acción a realizar esta dada por la política
            observation, reward, done, info = env.step(action)
            rewards_epi=rewards_epi+reward
            if render:
                env.render()
            time.sleep(0.0016)
            if done:
                if reward >= 0:
                    wins += 1
                print(f"Episode {i_episode} finished after {t+1} timesteps with reward {rewards_epi}")
                break
            t += 1
    time.sleep(0.0016)
    env.close()
    print("Victorias: ", wins)

#Q = qlearning(env, epsilon)
#Q = dQlearning(env, epsilon)
Q = lambdaQlearning(env, epsilon)
#Q = sarsa(env, epsilon)
#Q = lambdaSarsa(env, epsilon)

plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, rewards.__len__() + 1), rewards, marker='', color='b', label="Reward por episodio", linewidth=0.5)
plt.xlabel("Episodio")
plt.ylabel("Reward")
plt.title("Curva de aprendizaje: Reward vs episodio")
plt.legend()
plt.grid(True)
plt.savefig("4_map2_lambdaQlearning.png")  # Save the plot as a PNG image

playgames(env, Q, 100, True)
env.close()

#_ =env.step(env.action_space.sample())
