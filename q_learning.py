import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import pickle
import time
import os
from object import Object, show_sprites, get_unique_spawning_location

TILES = 10 #размер сетки 

ACTIONS = 4 #количество направлений движения (0, 1, 2, 3)
COUNT_EPISODES = 25000 #количество эпизодов обучения
MOVE_PUNISH = 1 #шаг
DOOM_PUNISH = 300 #столкновение с анти-целью
GOAL_REWARD = 25 #столкновение с целью
epsilon = 0.9
EPS_DECAY = 0.9998 #изменяем эпсилон
SHOW_EVERY = 2000  #демонстрация каждого SHOW_EVERY эпизода

LEARNING_RATE = 0.1
DISCOUNT = 0.95 #коэффициент дисконтирования

DISPLAY_SIZE = 300 #размер окна
SPRITE_SIZE = int(DISPLAY_SIZE / TILES)

#цвета спрайтов
BLUE = (232, 128, 58)
GREEN = (31, 166, 34)
RED = (58, 81, 232)

np.random.seed(1)

#создаем папку для таблицы в этой директории
if not os.path.isdir('tables'):
    os.makedirs('tables')

class QTable:#таблица пар состояние-действие 
    start_values = None#для обучения None, для демонстрации путь к файлу (tables\qtable...)
    q_table = {}
    tiles = [i for i in range(1 - TILES, TILES)]#кортеж значений (-5; 5)
    def __init__(self):
        if self.start_values is None:
            for x1 in tqdm(self.tiles):
                for y1 in self.tiles:
                    for x2 in self.tiles:
                        for y2 in self.tiles:
                            self.q_table[((x1, y1), (x2, y2))] = [np.random.uniform(1 - TILES , 0) for i in range(ACTIONS)]
        else: #если таблица уже существует, то считываем её
            with open(self.start_values, "r+b") as f:
                self.q_table = pickle.load(f)
    #по положению объектов определяем действие
    def choose_action(self, state):
        if np.random.random() > epsilon:
            return np.argmax(table.q_table[state]) #наиболее вероятное действие для достижения цели
        else:
            return np.random.randint(0, ACTIONS) #случайное


table = QTable()

episode_rewards = [] #очки за эпизод 

for episode in range(COUNT_EPISODES):
    #уникальное положение объектов
    dont_spawn_here = []

    agent = Object()
    dont_spawn_here.append((agent.x, agent.y))
    
    goal = get_unique_spawning_location(dont_spawn_here)
    dont_spawn_here.append((goal.x, goal.y))

    doom = get_unique_spawning_location(dont_spawn_here)
    
    #выводим в консоль среднее значение по эпизоду и эпсилон
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        state = (agent - goal, agent - doom) #положение объектов = состояния, по которым делается вывод
        #о дальнейших действиях 

        #принимаем решение о действии на основании состояний 
        action = table.choose_action(state)
        #совершаем это действие(action->move->agent.x, agent.y)
        agent.action(action)

        #изменяем очки, получаемые на данном шаге
        if agent == doom: #при столкновении с нежелательным объектом
            reward = -DOOM_PUNISH
        elif agent == goal: #и целью
            reward = GOAL_REWARD
        else:
            reward = -MOVE_PUNISH #за каждый шаг отнимаются очки

        #смотрим, как повлияло действие на положение агента относительно других объектов
        new_state = (agent - goal, agent - doom)
        max_future_q = np.max(table.q_table[new_state]) #maxQ(state(t+1), action), стремимся к наибольшим значениям 
        current_q = table.q_table[state][action] #предыдущее значение по таблице

        if reward == GOAL_REWARD: #при достижении цели
            new_q_value = GOAL_REWARD
        else:
            new_q_value = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q) #уравнение Беллмана
        table.q_table[state][action] = new_q_value #обновляем ей таблицу

        if show: 
            show_sprites(agent, goal, doom) #графические примитивы, чтобы проиллюстрировать обучение

            #окончание эпизода 
            if reward == GOAL_REWARD or reward == -DOOM_PUNISH: #если столкнулись с объектом
                if cv2.waitKey(200) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(60) & 0xFF == ord('q'):
                    break
            
        episode_reward += reward #подсчитываем очки 
        if reward == GOAL_REWARD or reward == -DOOM_PUNISH:
            break

    episode_rewards.append(episode_reward) #награда за шаг 
    #epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-EPS_DECAY * episode)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode = 'valid') #преобразуем массив с очками в более удобный вид

plt.plot([i for i in range(len(moving_avg))], moving_avg) #строим график
#подписываем оси
plt.ylabel(f"Reward {SHOW_EVERY}ma") #вознаграждение
plt.xlabel("episode") #номер эпизода
plt.show()

with open(f"tables/qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(table.q_table, f)
#создали таблицу, которая сохраняется в pickle