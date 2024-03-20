import numpy as np
import keyboard
mat = [[16,0,2048,0],[16,0,0,2048],[16,0,16,2048],[0,0,0,0]]
import random

import pygame


def show():
    st = "{0:<8}".format("#")
    for element in mat :
        print("{0:#^33}".format(""))
        for i in range(5):
            if (i == 2):
                for e in element:
                    st += "#"
                    st += "{0:^7}".format(str(e) if e != 0 else "")
                st += "#"
                print(st)
                st = "{0:<8}".format("#")
            else:
                for e in element:
                    st += "{0:<8}".format("#")
                print(st)
                if(i == 1):
                    st = ""
                else:
                    st= "{0:<8}".format("#")
    print("{0:#^33}".format(""))

def move_left():
    for i in range(4):
        for j in range(3):
            if mat[i][j] == 0 and mat[i][j+1] != 0:
                num = j
                while(num>=0):
                    if mat[i][num] == 0 and mat[i][num + 1] != 0:
                        mat[i][num] = mat[i][num+1]
                        mat[i][num+1] = 0
                    else :
                        if mat[i][num] == mat[i][num + 1] and mat[i][num] != 0:
                            mat[i][num] *= 2
                            mat[i][num + 1] = 0
                        break
                    num -= 1
            elif mat[i][j] == mat[i][j + 1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j+1] = 0

def move_right():
    for i in range(4):
        for j in range(3, 0, -1):
            if mat[i][j] == 0 and mat[i][j-1] != 0:
                num = j
                while(num<=3):
                    if mat[i][num] == 0 and mat[i][num-1] != 0:
                        mat[i][num] = mat[i][num-1]
                        mat[i][num-1] = 0
                    else:
                        if mat[i][num] == mat[i][num-1] and mat[i][num] != 0:
                            mat[i][num] *= 2
                            mat[i][num-1] = 0
                        break
                    num += 1
            elif mat[i][j] == mat[i][j - 1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j - 1] = 0

def move_up():
    for i in range(4):
        for j in range(3):
            if mat[j][i] == 0 and mat[j + 1][i] != 0:
                num = j
                while(num>=0):
                    if mat[num][i] == 0 and mat[num+ 1][i] != 0:
                        mat[num][i] = mat[num + 1][i]
                        mat[num + 1][i] = 0
                    else:
                        if mat[num][i] == mat[num + 1][i] and mat[num][i] != 0:
                            mat[num][i] *= 2
                            mat[num + 1][i] = 0
                        break
                    num -= 1
            elif mat[j][i] == mat[j + 1][i] and mat[j][i] != 0:
                mat[j][i] *= 2
                mat[j + 1][i] = 0

def move_down():
    for i in range(4):
        for j in range(3, 0, -1):
            if mat[j][i] == 0 and  mat[j - 1][i] != 0:
                num = j
                while(num<=3):
                    if mat[num][i] == 0 and  mat[num - 1][i] != 0:
                        mat[num][i] = mat[num - 1][i]
                        mat[num - 1][i] = 0
                    else:
                        if mat[num][i] == mat[num - 1][i] and mat[num][i] != 0:
                            mat[num][i] *= 2
                            mat[num - 1][i] = 0
                        break
                    num += 1
            if mat[j][i] == mat[j - 1][i] and mat[j][i] != 0:
                mat[j][i] *= 2
                mat[j - 1][i] = 0

def on_key_event(event):
    if event.event_type == keyboard.KEY_DOWN:
        if event.name == 'up':
            print('向上键被按下')
            move_up()
        elif event.name == 'down':
            print('向下键被按下')
            move_down()
        elif event.name == 'left':
            print('向左键被按下')
            move_left()
        elif event.name == 'right':
            print('向右键被按下')
            move_right()
    random_generate()
    show()

def random_generate():
    num = 0
    for element in mat:
        for ele in element:
            if ele == 0 :
                num += 1
    random_num = random.randint(0, num)
    print(random_num)
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j] == 0:
                random_num -= 1
                if random_num == 0:
                    mat[i][j] = 2
                    print('2')

def init():
    keyboard.on_press(on_key_event)
    keyboard.wait('ctrl+c')


if __name__ == '__main__':
    show()
    keyboard.on_press(on_key_event)
    keyboard.wait('ctrl+c')


