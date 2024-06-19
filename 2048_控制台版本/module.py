import numpy as np
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

class grid():
    def __init__(self,num_Mat):
        self.num_Mat = num_Mat

    def get_smoothness(self): #光滑度
        smoothness = 0
        for i in range(4):
            for j in range(4):
                if self.num_Mat[i][j] != 0:
                    value = np.log2(self.num_Mat[i][j])
                    if i + 1 < 4:  # 下方的邻居
                        if self.num_Mat[i + 1][j] != 0:
                            target_value = np.log2(self.num_Mat[i + 1][j])
                            smoothness -= abs(value - target_value)
                    if j + 1 < 4:  # 右方的邻居
                        if self.num_Mat[i][j + 1] != 0:
                            target_value = np.log2(self.num_Mat[i][j + 1])
                            smoothness -= abs(value - target_value)
        return smoothness

    def monotonicity2(self): #单调性
        totals = [0, 0, 0, 0]

        # 向上/向下 方向
        for x in range(4):
            current = 0
            next = current + 1
            while next < 4:
                while next < 4 and self.num_Mat[x][next] == 0:
                    next += 1
                if next >= 4:
                    next -= 1
                currentValue = np.log2(self.num_Mat[x][current]) if self.num_Mat[x][current] != 0 else 0
                nextValue = np.log2(self.num_Mat[x][next]) if self.num_Mat[x][next] != 0 else 0
                if currentValue > nextValue:
                    totals[0] += nextValue - currentValue
                elif nextValue > currentValue:
                    totals[1] += currentValue - nextValue
                current = next
                next += 1

        # 向左/向右 方向
        for y in range(4):
            current = 0
            next = current + 1
            while next < 4:
                while next < 4 and self.num_Mat[next][y] == 0:
                    next += 1
                if next >= 4:
                    next -= 1
                currentValue = np.log2(self.num_Mat[current][y]) if self.num_Mat[current][y] != 0 else 0
                nextValue = np.log2(self.num_Mat[next][y]) if self.num_Mat[next][y] != 0 else 0
                if currentValue > nextValue:
                    totals[2] += nextValue - currentValue
                elif nextValue > currentValue:
                    totals[3] += currentValue - nextValue
                current = next
                next += 1

        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    def maxValue(self):#获取最大值
        max_value = 0
        for x in range(4):
            for y in range(4):
                if self.num_Mat[x][y] != 0:
                    value = self.num_Mat[x][y]
                    if value > max_value:
                        max_value = value

        return np.log2(max_value)

    def numberOfEmptyCells(self):#计算0格子数量
        return list(self.num_Mat.flatten()).count(0)

    def evaluate_score(self):#计算格局评分

        # 标准权重
        smooth_weight = 0.1
        mono2_weight = 1.0
        empty_weight = 2.7
        max_weight = 1.0

        # 计算各项评分
        smoothness_score = self.smoothness() * smooth_weight
        monotonicity_score = self.monotonicity2() * mono2_weight
        emptiness_score = np.log(self.numberOfEmptyCells()) * empty_weight
        max_value_score = self.maxValue() * max_weight

        return smoothness_score + monotonicity_score + emptiness_score + max_value_score