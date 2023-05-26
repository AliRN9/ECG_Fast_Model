import cv2 as cv2
import glob
import numpy as np
from pathlib import Path
import re


def sort_by(file: str) -> int:
    # print(file)
    '''
    '''
    file = file.split('/')[-1]
    file = file.split('_')[0]
    return int(file)
    # print((re.search(r'\d+', file)[0]))
    # return int(re.search(r'\d{3,}', file)[0])

data = glob.glob('data_potential/*.png')
data.sort(key=sort_by)
# print(data)
#читаем кадры
# imgs = [cv2.imread(str(f)) for f in data ]
# #
# # # создаем видео
# height, width, layers = imgs[0].shape
# video = cv2.VideoWriter(r'd:/temp/video.avi',-1,1,(width,height))
# _ = [video.write(i) for i in imgs]
# #
# cv2.destroyAllWindows()
# video.release()

# Получение списка файлов в папке

# Установка размера видео
frame_size = (640, 480)

# Создание объекта VideoWriter
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, frame_size)

# Цикл по всем файлам в папке
for file in data:
    # Чтение изображения
    img = cv2.imread(file)
    # Изменение размера изображения
    img = cv2.resize(img, frame_size)
    # Запись кадра в видео
    out.write(img)

# Освобождение ресурсов
out.release()