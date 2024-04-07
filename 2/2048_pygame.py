import copy
import random
import sys
from core_2048 import init,move_up,move_down,move_left,move_right
import pygame

bg = (205,193,180)
class dot:
    x = 0
    y = 0
    def __init__(self,x,y,screen):
        self.x = x
        self.y = y
        self.screen = screen
class block:
    x = 0
    y = 0
    position = dot(0,0,screen=None)
    def __init__(self,position):
        self.position = position
        self.x = 100 + (self.position.x - 1) * 70
        self.y = 100 + (self.position.y - 1) * 70
    def update(self,txtsurf):
        pygame.draw.rect(self.position.screen, bg, pygame.Rect(self.x, self.y, 60, 60))
        if txtsurf is not None:
            text_x = self.x + (60 - txtsurf.get_width()) // 2
            text_y = self.y + (60 - txtsurf.get_height()) // 2
            self.position.screen.blit(txtsurf, (text_x, text_y))
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
        self.font =     pygame.font.Font(None, 36)
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

        # 找出最大数
        max_number = max(up_max, down_max, left_max, right_max)
        print(max_number)

        # 如果最大数在角落，则优先选择该方向
        if self.is_corner(up_max_position):
            return 'up'
        elif self.is_corner(down_max_position):
            return 'down'
        elif self.is_corner(left_max_position):
            return 'left'
        elif self.is_corner(right_max_position):
            return 'right'

        # 找出四个角落中最大数最大的方向
        corner_max = max(up_max, down_max, left_max, right_max)
        max_direction = None
        if corner_max == up_max:
            max_direction = 'up'
        elif corner_max == down_max:
            max_direction = 'down'
        elif corner_max == left_max:
            max_direction = 'left'
        elif corner_max == right_max:
            max_direction = 'right'

        # 检查每个方向移动后存在最少0的方向，并返回该方向
        min_zeros = float('inf')
        min_zeros_direction = None
        for direction, matrix in [('up', up_matrix), ('down', down_matrix), ('left', left_matrix),
                                  ('right', right_matrix)]:
            num_zeros = sum(row.count(0) for row in matrix)
            if num_zeros < min_zeros:
                min_zeros = num_zeros
                min_zeros_direction = direction

        # 如果有多个方向的最大数都在角落中，则选择其中最大最大数和最少0的方向
        if max_direction:
            return max_direction
        else:
            return min_zeros_direction

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
red = (255,0,0)
if __name__ == '__main__':
    game = gameManager()
    done = False
    while not done:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            # 检查按键事件
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.move_up(game.num_Mat)
                    game.random_generate()
                elif event.key == pygame.K_DOWN:
                    game.move_down(game.num_Mat)
                    game.random_generate()
                elif event.key == pygame.K_LEFT:
                    game.move_left(game.num_Mat)
                    game.random_generate()
                elif event.key == pygame.K_RIGHT:
                    game.move_right(game.num_Mat)
                    game.random_generate()
                print(game.predict_next_move())
        game.update()
    """while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                    # 检测上下左右键是否被按下
                    if event.key == pygame.K_UP:
                        move_up()
                    elif event.key == pygame.K_DOWN:
                        move_down()
                    elif event.key == pygame.K_LEFT:
                        move_left()
                    elif event.key == pygame.K_RIGHT:
                        move_right()"""