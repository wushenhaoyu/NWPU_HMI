import copy

class Advice:
    def __init__(self, num_Mat):
        self.num_Mat = num_Mat

    def predict_next_move(self):
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
            return self.find_max_zeros_direction(up_matrix, down_matrix, left_matrix, right_matrix)

        max_corner = max(corners, key=lambda x: x[1])
        max_directions = [corner for corner in corners if corner[1] == max_corner[1]]

        if len(max_directions) == 1:
            return max_directions[0][0]

        max_zeros = float('-inf')
        max_zeros_direction = None
        for direction, _, matrix in max_directions:
            num_zeros = sum(row.count(0) for row in matrix)
            if num_zeros > max_zeros:
                max_zeros = num_zeros
                max_zeros_direction = direction

        return max_zeros_direction

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
        return position == (0, 0) or position == (0, 3) or position == (3, 0) or position == (3, 3)

    # 下面这些函数可以根据你的具体需求自行实现
    def move_up(self, matrix):
        return matrix.move_up()

    def move_down(self, matrix):
        return matrix.move_down

    def move_left(self, matrix):
        pass

    def move_right(self, matrix):
        pass
