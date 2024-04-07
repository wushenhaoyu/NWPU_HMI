import numpy as np
import copy
import random

class mat:
    def __init__(self, game_matrix):
        self.game_matrix = game_matrix
        self.size = game_matrix.shape[0]

    def __copy__(self):
        # 创建一个新的对象，并使用浅复制来复制属性
        new_game_matrix = copy.copy(self.game_matrix)
        new_size = self.size
        new_obj = mat(new_game_matrix)
        new_obj.size = new_size
        return new_obj

    def __deepcopy__(self, memo):
        # 创建一个新的对象，并使用深复制来复制属性
        new_game_matrix = copy.deepcopy(self.game_matrix, memo)
        new_size = self.size
        new_obj = mat(new_game_matrix)
        new_obj.size = new_size
        return new_obj

    def monotonicity(self):
        # 单调性
        total_score = 0
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.game_matrix[i][j] <= self.game_matrix[i][j + 1]:
                    total_score += 1
                if self.game_matrix[j][i] <= self.game_matrix[j + 1][i]:
                    total_score += 1
        return total_score

    def smoothness(self):
        # 平滑性
        total_score = 0
        for i in range(self.size):
            for j in range(self.size):
                if i > 0:
                    total_score -= abs(self.game_matrix[i][j] - self.game_matrix[i - 1][j])
                if i < self.size - 1:
                    total_score -= abs(self.game_matrix[i][j] - self.game_matrix[i + 1][j])
                if j > 0:
                    total_score -= abs(self.game_matrix[i][j] - self.game_matrix[i][j - 1])
                if j < self.size - 1:
                    total_score -= abs(self.game_matrix[i][j] - self.game_matrix[i][j + 1])
        return total_score

    def empty_tiles(self):
        # 空格数
        return np.sum(self.game_matrix == 0)

    def max_value(self):
        # 最大数
        return np.max(self.game_matrix)

    def move_left(self):
        # 向左移动
        new_game_matrix = np.zeros_like(self.game_matrix)
        for i in range(self.size):
            row = self.game_matrix[i]
            row = self._collapse_row(row)
            new_game_matrix[i] = row
        return new_game_matrix

    def move_right(self):
        # 向右移动
        new_game_matrix = np.zeros_like(self.game_matrix)
        for i in range(self.size):
            row = self.game_matrix[i][::-1]
            row = self._collapse_row(row)
            new_game_matrix[i] = row[::-1]
        return new_game_matrix

    def move_up(self):
        # 向上移动
        new_game_matrix = np.zeros_like(self.game_matrix)
        for j in range(self.size):
            col = self.game_matrix[:, j]
            col = self._collapse_row(col)
            new_game_matrix[:, j] = col
        return new_game_matrix

    def move_down(self):
        # 向下移动
        new_game_matrix = np.zeros_like(self.game_matrix)
        for j in range(self.size):
            col = self.game_matrix[::-1, j]
            col = self._collapse_row(col)
            new_game_matrix[:, j] = col[::-1]
        return new_game_matrix

    def move(self,direction):
        if direction == 0 :
            return self.move_up()
        elif direction == 1:
            return self.move_down()
        elif direction == 2:
            return self.move_left()
        elif direction == 3:
            return self.move_right()

    def compare(self,a):
        for i in range(4):
            for j in range(4):
                if self.game_matrix[i][j] != a.game_matrix[i][j]:
                    return False
        return True


    def _collapse_row(self, row):
        new_row = np.zeros_like(row)
        idx = 0
        for i in range(len(row)):
            if row[i] != 0:
                if new_row[idx] == row[i]:
                    new_row[idx] *= 2
                    idx += 1
                elif new_row[idx] == 0:
                    new_row[idx] = row[i]
                else:
                    idx += 1
                    new_row[idx] = row[i]
        return new_row

    def generate_random_tile(self):
        # 在空格上生成随机数字方块
        empty_indices = np.argwhere(self.game_matrix == 0)
        if len(empty_indices) > 0:
            random_index = random.choice(empty_indices)
            self.game_matrix[random_index[0]][random_index[1]] = 2 if random.random() < 0.9 else 4

    def evaluation(self):
        empty_cells = np.sum(self.game_matrix == 0)

        smooth_weight = 0.1
        mono_weight = 1.0
        empty_weight = 2.7
        max_weight = 1.0

        smoothness_score = self.smoothness() * smooth_weight
        mono2_score = self.monotonicity() * mono_weight
        empty_score = np.log(empty_cells) * empty_weight
        max_score = np.max(self.game_matrix) * max_weight

        total_score = smoothness_score + mono2_score + empty_score + max_score
        return total_score

    def availableCells(self):
        cells = []
        for i in range(4):
            for j in range(4):
                if self.game_matrix[i][j] == 0:
                    cells.append((i, j))
        return cells

    def insert(self,dot,value):
        self.game_matrix[dot[0]][dot[1]] = value

    def remove(self,dot,value):
        self.game_matrix[dot[0]][dot[1]] = 0

    def isWin(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.game_matrix[i][j] == 2048:
                    return True
        return False



