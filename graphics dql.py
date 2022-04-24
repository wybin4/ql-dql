import json
import matplotlib.pyplot as plt
import numpy as np

# открываем файл в режиме чтения
with open('rewards 500.txt', 'r') as fr:
    # читаем из файла
    rewards = json.load(fr)
    
moving_avg = np.convolve(rewards, np.ones((5,))/5, mode = 'valid') #преобразуем массив с очками в более удобный вид

plt.plot([i for i in range(len(moving_avg))], moving_avg) #строим график
#подписываем оси
plt.ylabel(f"Reward") #вознаграждение
plt.xlabel("episode") #номер эпизода
plt.show()
