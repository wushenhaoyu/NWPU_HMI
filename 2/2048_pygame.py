import copy
import random

import numpy as np
import pygame

from module import dot, block


bg = (205,193,180)

class gameManager:

    block_Mat = []
    num_Mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        self.screen.fill((250,248,239))
        self.screen.fill((187,173,160),rect=(90,90,290,290))
        for i in range(0,4):
            newLine = []
            for j in range(0,4):
                newLine.append(block(dot(j+1,i+1,self.screen)))

            self.block_Mat.append(newLine)
        self.font = pygame.font.Font(None, 36)
        self.text1 = self.font.render("advice", True, (187,173,160))
        self.rect1 = self.text1.get_rect(topleft=(10,10))
        self.screen.blit(self.text1,self.rect1)
    def update(self):
        for i in range(0,4):
            for j in range(0,4):
                if(self.num_Mat[i][j] != 0):
                    txtsurf = self.font.render(str(self.num_Mat[i][j]), True, (119,110,101))
                    self.block_Mat[i][j].update(txtsurf)
                else:
                    self.block_Mat[i][j].update(None)

        pygame.display.update()

    def random_generate(self):
        num = 0
        for element in self.num_Mat:
            for ele in element:
                if ele == 0:
                    num += 1
        random_num = random.randint(0, num)
        for i in range(len(self.num_Mat)):
            for j in range(len(self.num_Mat[i])):
                if self.num_Mat[i][j] == 0:
                    random_num -= 1
                    if random_num == 0:
                        self.num_Mat[i][j] = 2

    def move_left(self, num_Mat):
        for i in range(4):
            for j in range(3):
                if num_Mat[i][j] == 0 and num_Mat[i][j + 1] != 0:
                    num = j
                    while (num >= 0):
                        if num_Mat[i][num] == 0 and num_Mat[i][num + 1] != 0:
                            num_Mat[i][num] = num_Mat[i][num + 1]
                            num_Mat[i][num + 1] = 0
                        else:
                            if num_Mat[i][num] == num_Mat[i][num + 1] and num_Mat[i][num] != 0:
                                num_Mat[i][num] *= 2
                                num_Mat[i][num + 1] = 0
                            break
                        num -= 1
                elif num_Mat[i][j] == num_Mat[i][j + 1] and num_Mat[i][j] != 0:
                    num_Mat[i][j] *= 2
                    num_Mat[i][j + 1] = 0
        return num_Mat

    def move_right(self, num_Mat):
        for i in range(4):
            for j in range(3, 0, -1):
                if num_Mat[i][j] == 0 and num_Mat[i][j - 1] != 0:
                    num = j
                    while (num <= 3):
                        if num_Mat[i][num] == 0 and num_Mat[i][num - 1] != 0:
                            num_Mat[i][num] = num_Mat[i][num - 1]
                            num_Mat[i][num - 1] = 0
                        else:
                            if num_Mat[i][num] == num_Mat[i][num - 1] and num_Mat[i][num] != 0:
                                num_Mat[i][num] *= 2
                                num_Mat[i][num - 1] = 0
                            break
                        num += 1
                elif num_Mat[i][j] == num_Mat[i][j - 1] and num_Mat[i][j] != 0:
                    num_Mat[i][j] *= 2
                    num_Mat[i][j - 1] = 0
        return num_Mat

    def move_up(self, num_Mat):
        for i in range(4):
            for j in range(3):
                if num_Mat[j][i] == 0 and num_Mat[j + 1][i] != 0:
                    num = j
                    while (num >= 0):
                        if num_Mat[num][i] == 0 and num_Mat[num + 1][i] != 0:
                            num_Mat[num][i] = num_Mat[num + 1][i]
                            num_Mat[num + 1][i] = 0
                        else:
                            if num_Mat[num][i] == num_Mat[num + 1][i] and num_Mat[num][i] != 0:
                                num_Mat[num][i] *= 2
                                num_Mat[num + 1][i] = 0
                            break
                        num -= 1
                elif num_Mat[j][i] == num_Mat[j + 1][i] and num_Mat[j][i] != 0:
                    num_Mat[j][i] *= 2
                    num_Mat[j + 1][i] = 0
        return num_Mat

    def move_down(self, num_Mat):
        for i in range(4):
            for j in range(3, 0, -1):
                if num_Mat[j][i] == 0 and num_Mat[j - 1][i] != 0:
                    num = j
                    while (num <= 3):
                        if num_Mat[num][i] == 0 and num_Mat[num - 1][i] != 0:
                            num_Mat[num][i] = num_Mat[num - 1][i]
                            num_Mat[num - 1][i] = 0
                        else:
                            if num_Mat[num][i] == num_Mat[num - 1][i] and num_Mat[num][i] != 0:
                                num_Mat[num][i] *= 2
                                num_Mat[num - 1][i] = 0
                            break
                        num += 1
                if num_Mat[j][i] == num_Mat[j - 1][i] and num_Mat[j][i] != 0:
                    num_Mat[j][i] *= 2
                    num_Mat[j - 1][i] = 0
        return num_Mat

    def move(self,direction,mat):#上下左右
        if direction == 0:
            return self.move_up(mat)
        elif direction == 1:
            return self.move_down(mat)
        elif direction == 2:
            return self.move_left(mat)
        elif direction == 3:
            return self.move_right(mat)

    def compareMat(self,a,b):
        a = np.array(a)
        b = np.array(b)
        return np.array_equal(a, b)


    # 添加预测下一步移动方向的方法
    def predict_next_move(self):
        # 计算向上、向下、向左、向右移动后的矩阵，并记录每个方向的最大数和最大数位置
        up_matrix = self.move_up(copy.deepcopy(self.num_Mat))
        down_matrix = self.move_down(copy.deepcopy(self.num_Mat))
        left_matrix = self.move_left(copy.deepcopy(self.num_Mat))
        right_matrix = self.move_right(copy.deepcopy(self.num_Mat))

        up_max = self.get_max_number(up_matrix)
        down_max = self.get_max_number(down_matrix)
        left_max = self.get_max_number(left_matrix)
        right_max = self.get_max_number(right_matrix)

        up_max_position = self.get_max_position(up_matrix)
        down_max_position = self.get_max_position(down_matrix)
        left_max_position = self.get_max_position(left_matrix)
        right_max_position = self.get_max_position(right_matrix)

        # 第一次筛选：找出最大数在角落的方向
        corners = []
        if self.is_corner(up_max_position):
            corners.append(('up', up_max, up_matrix))
        if self.is_corner(down_max_position):
            corners.append(('down', down_max, down_matrix))
        if self.is_corner(left_max_position):
            corners.append(('left', left_max, left_matrix))
        if self.is_corner(right_max_position):
            corners.append(('right', right_max, right_matrix))

        if not corners:
            return self.find_max_zeros_direction(up_matrix, down_matrix, left_matrix,
                                                 right_matrix)  # 如果没有最大数在角落，返回含0最多的方向

        # 第二次筛选：在第一次筛选的基础上找出最大数在角落的最大的方向
        max_corner = max(corners, key=lambda x: x[1])
        max_directions = [corner for corner in corners if corner[1] == max_corner[1]]

        if len(max_directions) == 1:
            return max_directions[0][0]  # 如果只有一个方向，直接返回该方向

        # 第三次筛选：在第二次筛选的基础上，找出矩阵中0数量最多的方向
        max_zeros = float('-inf')
        max_zeros_direction = None
        for direction, _, matrix in max_directions:
            num_zeros = sum(row.count(0) for row in matrix)
            if num_zeros > max_zeros:
                max_zeros = num_zeros
                max_zeros_direction = direction

        return max_zeros_direction  # 返回第三次筛选出的方向

    def find_max_zeros_direction(self, *matrices):
        max_zeros = float('-inf')
        max_zeros_direction = None
        for direction, matrix in zip(['up', 'down', 'left', 'right'], matrices):
            num_zeros = sum(row.count(0) for row in matrix)
            if num_zeros > max_zeros:
                max_zeros = num_zeros
                max_zeros_direction = direction
        return max_zeros_direction

    def get_max_number(self, matrix):
        max_number = 0
        for row in matrix:
            max_number = max(max_number, max(row))
        return max_number

    def get_max_position(self, matrix):
        max_number = self.get_max_number(matrix)
        for i in range(4):
            for j in range(4):
                if matrix[i][j] == max_number:
                    return (i, j)

    def is_corner(self, position):
        # 检查位置是否在矩阵的四角
        if position == (0, 0) or position == (0, 3) or position == (3, 0) or position == (3, 3):
            return True
        else:
            return False


    def search(self,grid,depth,alpha,beta,positions,cutoffs):
        bestMove = -1
        for direction in range(4):
            newMat = copy.deepcopy(self.num_Mat)
            self.move(direction,newMat)







red = (255, 0, 0)

red = (255, 0, 0)

red = (255, 0, 0)

if __name__ == '__main__':
    game = gameManager()
    done = False
    last_move_direction = None  # 存储上一次移动的方向
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.move_up(game.num_Mat)
                    game.random_generate()
                    last_move_direction = 'up'  # 更新上一次移动的方向
                elif event.key == pygame.K_DOWN:
                    game.move_down(game.num_Mat)
                    game.random_generate()
                    last_move_direction = 'down'  # 更新上一次移动的方向
                elif event.key == pygame.K_LEFT:
                    game.move_left(game.num_Mat)
                    game.random_generate()
                    last_move_direction = 'left'  # 更新上一次移动的方向
                elif event.key == pygame.K_RIGHT:
                    game.move_right(game.num_Mat)
                    game.random_generate()
                    last_move_direction = 'right'  # 更新上一次移动的方向
                game.screen.fill((250, 248, 239), (0, 40, 100, 40))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position
                position = pygame.mouse.get_pos()

                # Check if mouse position is on the "advice" text
                if game.rect1.collidepoint(position):
                    if last_move_direction:
                        print("提示：", game.predict_next_move())
                        text_surface = game.font.render(game.predict_next_move(), True, (187, 173, 160))
                        text_rect = text_surface.get_rect(center=(40, 60))
                        game.screen.blit(text_surface, text_rect)


        game.update()