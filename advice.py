import copy


class MatrixAdvisor:
    def __init__(self):
        self.matrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    def update(self,m):
        self.matrix = m


    def predict_next_move(self):
        up_matrix = self.move_up(copy.deepcopy(self.matrix))
        down_matrix = self.move_down(copy.deepcopy(self.matrix))
        left_matrix = self.move_left(copy.deepcopy(self.matrix))
        right_matrix = self.move_right(copy.deepcopy(self.matrix))

        directions = {
            'up': (up_matrix, up_matrix),
            'down': (down_matrix, down_matrix),
            'left': (left_matrix, left_matrix),
            'right': (right_matrix, right_matrix)
        }

        best_direction = max(directions.items(), key=lambda item: self.get_max_number(item[1][0]))
        return best_direction[0]

    def move_up(self, matrix):
        for col in range(4):
            stack = []
            for row in range(4):
                if matrix[row][col]:
                    stack.append(matrix[row][col])
            new_col = [0] * 4
            for i in range(len(stack)):
                if i + 1 < len(stack) and stack[i] == stack[i + 1]:
                    new_col[i] = stack[i] * 2
                    stack[i + 1] = 0
                else:
                    new_col[i] = stack[i]
            for row in range(4):
                matrix[row][col] = new_col[row]
        return matrix

    def move_down(self, matrix):
        for col in range(4):
            stack = []
            for row in reversed(range(4)):
                if matrix[row][col]:
                    stack.append(matrix[row][col])
            new_col = [0] * 4
            for i in range(len(stack)):
                if i + 1 < len(stack) and stack[i] == stack[i + 1]:
                    new_col[3 - i] = stack[i] * 2
                    stack[i + 1] = 0
                else:
                    new_col[3 - i] = stack[i]
            for row in reversed(range(4)):
                matrix[row][col] = new_col[3 - row]
        return matrix

    def move_left(self, matrix):
        for row in range(4):
            stack = []
            for col in range(4):
                if matrix[row][col]:
                    stack.append(matrix[row][col])
            new_row = [0] * 4
            for i in range(len(stack)):
                if i + 1 < len(stack) and stack[i] == stack[i + 1]:
                    new_row[i] = stack[i] * 2
                    stack[i + 1] = 0
                else:
                    new_row[i] = stack[i]
            for col in range(4):
                matrix[row][col] = new_row[col]
        return matrix

    def move_right(self, matrix):
        for row in range(4):
            stack = []
            for col in reversed(range(4)):
                if matrix[row][col]:
                    stack.append(matrix[row][col])
            new_row = [0] * 4
            for i in range(len(stack)):
                if i + 1 < len(stack) and stack[i] == stack[i + 1]:
                    new_row[3 - i] = stack[i] * 2
                    stack[i + 1] = 0
                else:
                    new_row[3 - i] = stack[i]
            for col in reversed(range(4)):
                matrix[row][col] = new_row[3 - col]
        return matrix

    def get_max_number(self, matrix):
        return max(map(max, matrix))
