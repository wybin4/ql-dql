import numpy as np
from PIL import Image
import cv2

TILES = 10 #размер сетки 
DISPLAY_SIZE = 300 #размер окна
SPRITE_SIZE = int(DISPLAY_SIZE / TILES)
#цвета
BLUE = (232, 128, 58)
GREEN = (31, 166, 34)
RED = (58, 81, 232)
RANDOM_SEED = 0

class Object:
    def __init__(self):#содержит случайные координаты
        self.x = np.random.randint(0, TILES)
        self.y = np.random.randint(0, TILES)

    def __sub__(self, other): #вычитание объектов класса
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other): #сравнение объектов класса
        return self.x == other.x and self.y == other.y

    def action(self, choice): #возможные действия по поданному параметру 
        #(0, 1, 2, 3)
        if choice == 0:
            self.move(x = 1, y = 1)
        elif choice == 1:
            self.move(x = -1, y = -1)
        elif choice == 2:
            self.move(x = -1, y = 1)
        elif choice == 3:
            self.move(x = 1, y = -1)

    def move(self, x = False, y = False): #реализует перемещение по координатам объекта

        #если значение х не было подано, то оно генерируется случайным образом 
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        #если значение у не было подано, то оно генерируется случайным образом 
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        #если вышли за границы окна, то возвращаемся обратно
        if self.x < 0:
            self.x = 0
        elif self.x > TILES - 1:
            self.x = TILES - 1
        if self.y < 0:
            self.y = 0
        elif self.y > TILES - 1:
            self.y = TILES - 1

#исключаем случаи совпадения цели и агента с самого начала 
def get_unique_spawning_location(dont_spawn_here):
    all_x, all_y = zip(*dont_spawn_here)
    object = Object()
    
    while (object.x in all_x) and (object.y in all_y):
        object = Object() #создаем объекты заново с другими координатами
    
    return object

def show_sprites(agent, goal, doom): #обновляем картинку
    env = np.zeros((300, 300, 3), dtype = np.uint8)

    env [(agent.x) * SPRITE_SIZE: (agent.x + 1) * SPRITE_SIZE, (agent.y) * SPRITE_SIZE:(agent.y + 1) * SPRITE_SIZE, :] = BLUE #квадрат агента
    env [(goal.x) * SPRITE_SIZE:(goal.x + 1) * SPRITE_SIZE, (goal.y) * SPRITE_SIZE:(goal.y + 1) * SPRITE_SIZE, :] = GREEN #квадрат цели
    env [(doom.x) * SPRITE_SIZE:(doom.x + 1) * SPRITE_SIZE, (doom.y) * SPRITE_SIZE:(doom.y + 1) * SPRITE_SIZE, :] = RED #квадрат анти-цели
    img = Image.fromarray(env, 'RGB')

    img = Image.fromarray(env, 'RGB') #img-объект и перевод из bgr -> rgb

    cv2.imshow("", np.array(img)) #воспроизводим картинку 
    cv2.waitKey(1)