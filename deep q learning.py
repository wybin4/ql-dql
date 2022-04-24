import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from collections import deque
import random
import os
import time
import cv2
import matplotlib.pyplot as plt
from object import Object, show_sprites, get_unique_spawning_location

TILES = 10 #размер сетки 
DISPLAY_SIZE = 300 #размер окна
SPRITE_SIZE = int(DISPLAY_SIZE / TILES)
SHOW_PREVIEW = True

RANDOM_SEED = 0
MODEL_NAME = 'model'

MOVE_PUNISH = 1 #шаг
DOOM_PUNISH = 300 #столкновение с анти-целью
GOAL_REWARD = 25 #столкновение с целью
ACTIONS = 4 #количество направлений движения (0, 1, 2, 3)

MINI_BATCH_SIZE = 64 * 2 #размер батча
LEARNING_RATE = 0.001
DISCOUNT = 0.618 #коэффициент дисконтирования
OBSERVATION_VALUES = (4) #пространство состояний
epsilon = 0.9
EPS_DECAY = 0.9998 #для изменения эпсилон
REPLAY_MEMORY_SIZE = 50_000  #размер области данных
UPDATE_TARGET_EVERY = 100 #обновляем веса текущей нс каждые 
TRAIN_EPISODES = 300 #количество эпизодов обучения

#создаем папку для модели в этой директории
if not os.path.isdir('models'):
    os.makedirs('models')

tf.random.set_seed(RANDOM_SEED) 

#создание модели нейронной сети
def create_model(states, actions):
    model = Sequential()
    #определяет способ установки начальных весов
    init = tf.keras.initializers.HeUniform()        
    #входной слой с 24 нейронами, акт функция relu
    model.add(Dense(24, input_dim = states, activation = 'relu', kernel_initializer = init))
    #скрытый слой с 12 нейронами, акт функция relu
    model.add(Dense(12, activation = 'relu', kernel_initializer = init))
    #выходной слой, акт функция linear
    model.add(Dense(actions, activation = 'linear', kernel_initializer = init))
    #собираем модель, потери - функция потерь хьюбера, оптимизатор адам
    model.compile(loss = tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), metrics = ['accuracy'])
    return model

def train(replay_memory, q_model, target_model, done):

    MIN_REPLAY_SIZE = 1000
    #пока не сохранили нужное количество, не начинаем обучение
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    #случайная выборка из памяти
    minibatch = random.sample(replay_memory, MINI_BATCH_SIZE)

    #получив состояние среды, подаем его в модель, чтобы получить q-значения
    current_states = np.array([transition[0] for transition in minibatch])
    current_qs_list = q_model.predict(current_states)

    #получаем будущее состояние системы и снова используем модель для q-значений
    new_current_states = np.array([transition[3] for transition in minibatch])
    future_qs_list = target_model.predict(new_current_states)

    #тренировочные данные
    X = []
    y = []

    for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

        #если не конец эпизода, то получаем q-значение 
        #и по уравнению Беллмана вычисляем новое q
        if not done:
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + DISCOUNT * max_future_q
        else:
            new_q = reward

        #обновляем q-значение для этого состояния
        current_qs = current_qs_list[index]
        current_qs[action] = new_q

        #и добавляем к тренировочным данным 
        X.append(current_state)
        y.append(current_qs)

        #собираем модель
    q_model.fit(np.array(X), np.array(y), batch_size = MINI_BATCH_SIZE, verbose = 0, shuffle = False)

def main():
    epsilon = 0.9

    q_model = create_model(OBSERVATION_VALUES, ACTIONS)

    target_model = create_model(OBSERVATION_VALUES, ACTIONS)
    target_model.set_weights(q_model.get_weights())

    replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

    dont_spawn_here = []

    agent = Object()
    dont_spawn_here.append((agent.x, agent.y))
        
    goal = get_unique_spawning_location(dont_spawn_here)
    dont_spawn_here.append((goal.x, goal.y))

    doom = get_unique_spawning_location(dont_spawn_here)

    steps_to_update_target_model = 0
    all_episodes_reward = []
    for episode in range(TRAIN_EPISODES):
        episode_step = 0
        total_training_rewards = 0
        state = np.array((agent - goal) + (agent - doom))
        done = False
        while not done:
            steps_to_update_target_model += 1
            random_number = np.random.rand()
            if random_number <= epsilon:
                action = np.random.randint(0, ACTIONS)
            else:
                predicted = q_model.predict(np.array(state).reshape(-1, *state.shape))[0]
                action = np.argmax(predicted)
            agent.action(action)
            new_state = np.array((agent - goal) + (agent - doom)) #состояние среды по сумме координат
            #изменяем очки, получаемые на данном шаге
            if agent == doom: #при столкновении с нежелательным объектом
                reward = -DOOM_PUNISH
            elif agent == goal: #и целью
                reward = GOAL_REWARD
            else:
                reward = -MOVE_PUNISH #за каждый шаг отнимаются очки
            episode_step += 1
            done = False #индикатор окончания эпизода
            if reward == GOAL_REWARD or reward == -DOOM_PUNISH or episode_step >= 200:
                done = True
                if cv2.waitKey(200) & 0xFF == ord('q'):
                    break
                else:
                    if cv2.waitKey(60) & 0xFF == ord('q'):
                        break
            replay_memory.append([state, action, reward, new_state, done])

            if steps_to_update_target_model % 4 == 0 or done:
                train(replay_memory, q_model, target_model, done)

            state = new_state
            total_training_rewards += reward

            if SHOW_PREVIEW and not episode % 1:
                show_sprites(agent, goal, doom)

            if done:
                total_training_rewards += 1

                if steps_to_update_target_model >= UPDATE_TARGET_EVERY:
                    target_model.set_weights(q_model.get_weights())
                    steps_to_update_target_model = 0
                break
        epsilon *= EPS_DECAY
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"ep mean: {np.mean(total_training_rewards)}")        
        all_episodes_reward.append(total_training_rewards)
        q_model.save(f'models/{MODEL_NAME}__{np.mean(all_episodes_reward[-25:])}mean__{int(time.time())}.model')
    return all_episodes_reward

if __name__ == '__main__':
    episode_rewards = main()
    moving_avg = np.convolve(episode_rewards, np.ones((5,))/5, mode = 'valid') #преобразуем массив с очками в более удобный вид

    plt.plot([i for i in range(len(moving_avg))], moving_avg) #строим график
    #подписываем оси
    plt.ylabel(f"Reward") #вознаграждение
    plt.xlabel("episode") #номер эпизода
    plt.show()
