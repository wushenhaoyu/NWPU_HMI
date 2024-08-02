import numpy as np
import pygame
import random
import math
from ai import  AI
from functional import mat
from advice import  MatrixAdvisor
pygame.init()

FPS = 60
font = pygame.font.Font(None, 80)

font1 = pygame.font.Font(None, 60)
class Tile:
    COLORS = [
        (237, 229, 218),
        (238, 225, 201),
        (243, 178, 122),
        (246, 150, 101),
        (247, 124, 95),
        (247, 95, 59),
        (237, 208, 115),
        (237, 204, 99),
        (236, 202, 80),
    ]


    def __init__(self, value, row, col, rect_width, rect_height):
        self.value = value
        self.row = row
        self.col = col
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.x = col * self.rect_width
        self.y = row * self.rect_height

    def get_color(self):
        color_index = int(math.log2(self.value)) - 1
        color = self.COLORS[color_index]
        return color

    def draw(self, window):
        color = self.get_color()
        pygame.draw.rect(window, color, (self.x, self.y, self.rect_width, self.rect_height))

        font = pygame.font.SysFont("comicsans", 60, bold=True)
        text = font.render(str(self.value), 1, (119, 110, 101))
        window.blit(
            text,
            (
                self.x + (self.rect_width / 2 - text.get_width() / 2),
                self.y + (self.rect_height / 2 - text.get_height() / 2),
            ),
        )

    def set_pos(self, ceil=False):
        if ceil:
            self.row = math.ceil(self.y / self.rect_height)
            self.col = math.ceil(self.x / self.rect_width)
        else:
            self.row = math.floor(self.y / self.rect_height)
            self.col = math.floor(self.x / self.rect_width)

    def move(self, delta):
        self.x += delta[0]
        self.y += delta[1]
class Game2048:
    def __init__(self,smooth_weight,mono_weight,empty_weight,max_weight):
        self.WIDTH, self.HEIGHT = 800, 800
        self.ROWS, self.COLS = 4, 4
        self.RECT_HEIGHT = self.HEIGHT // self.ROWS
        self.RECT_WIDTH = self.WIDTH // self.COLS
        self.OUTLINE_COLOR = (187, 173, 160)
        self.OUTLINE_THICKNESS = 10
        self.BACKGROUND_COLOR = (205, 192, 180)
        self.FONT_COLOR = (119, 110, 101)
        self.MOVE_VEL = 20
        self.WINDOW = pygame.display.set_mode((1050, 800))
        pygame.display.set_caption("2048")
        self.tiles = {}  # Initialize tiles dictionary
        self.generate_tiles()
        self.ai = AI(mat(self.matrix_to_np(), smooth_weight,mono_weight,empty_weight,max_weight))
        self.ai_active = False
        self.advice_ = MatrixAdvisor()
        self.isGerenateAdvice = False
        self.smooth_weight = smooth_weight
        self.mono_weight = mono_weight
        self.empty_weight = empty_weight
        self.max_weight = max_weight
        self.grade = 0

    def draw_grid(self):
        for row in range(1, self.ROWS):
            y = row * self.RECT_HEIGHT
            pygame.draw.line(self.WINDOW, self.OUTLINE_COLOR, (0, y), (self.WIDTH, y), self.OUTLINE_THICKNESS)

        for col in range(1, self.COLS):
            x = col * self.RECT_WIDTH
            pygame.draw.line(self.WINDOW, self.OUTLINE_COLOR, (x, 0), (x, self.HEIGHT), self.OUTLINE_THICKNESS)

        pygame.draw.rect(self.WINDOW, self.OUTLINE_COLOR, (0, 0, self.WIDTH, self.HEIGHT), self.OUTLINE_THICKNESS)

    def draw(self):
        self.WINDOW.fill(self.BACKGROUND_COLOR)
        self.generate_button()
        if self.isGerenateAdvice:
            self.generate_advice()
        for tile in self.tiles.values():
            tile.draw(self.WINDOW)
        self.generate_AI_status()

        self.draw_grid()
        pygame.display.update()

    def get_random_pos(self):
        row = None
        col = None
        while True:
            row = random.randrange(0, self.ROWS)
            col = random.randrange(0, self.COLS)

            if f"{row}{col}" not in self.tiles:
                break

        return row, col

    def move_tiles(self, direction):
        updated = True
        blocks = set()
        self.isGerenateAdvice = False
        if direction == "left":
            sort_func = lambda x: x.col
            reverse = False
            delta = (-self.MOVE_VEL, 0)
            boundary_check = lambda tile: tile.col == 0
            get_next_tile = lambda tile: self.tiles.get(f"{tile.row}{tile.col - 1}")
            merge_check = lambda tile, next_tile: tile.x > next_tile.x + self.MOVE_VEL
            move_check = (
                lambda tile, next_tile: tile.x > next_tile.x + self.RECT_WIDTH + self.MOVE_VEL
            )
            ceil = True
        elif direction == "right":
            sort_func = lambda x: x.col
            reverse = True
            delta = (self.MOVE_VEL, 0)
            boundary_check = lambda tile: tile.col == self.COLS - 1
            get_next_tile = lambda tile: self.tiles.get(f"{tile.row}{tile.col + 1}")
            merge_check = lambda tile, next_tile: tile.x < next_tile.x - self.MOVE_VEL
            move_check = (
                lambda tile, next_tile: tile.x + self.RECT_WIDTH + self.MOVE_VEL < next_tile.x
            )
            ceil = False
        elif direction == "up":
            sort_func = lambda x: x.row
            reverse = False
            delta = (0, -self.MOVE_VEL)
            boundary_check = lambda tile: tile.row == 0
            get_next_tile = lambda tile: self.tiles.get(f"{tile.row - 1}{tile.col}")
            merge_check = lambda tile, next_tile: tile.y > next_tile.y + self.MOVE_VEL
            move_check = (
                lambda tile, next_tile: tile.y > next_tile.y + self.RECT_HEIGHT + self.MOVE_VEL
            )
            ceil = True
        elif direction == "down":
            sort_func = lambda x: x.row
            reverse = True
            delta = (0, self.MOVE_VEL)
            boundary_check = lambda tile: tile.row == self.ROWS - 1
            get_next_tile = lambda tile: self.tiles.get(f"{tile.row + 1}{tile.col}")
            merge_check = lambda tile, next_tile: tile.y < next_tile.y - self.MOVE_VEL
            move_check = (
                lambda tile, next_tile: tile.y + self.RECT_HEIGHT + self.MOVE_VEL < next_tile.y
            )
            ceil = False

        while updated:
            updated = False
            sorted_tiles = sorted(self.tiles.values(), key=sort_func, reverse=reverse)

            for i, tile in enumerate(sorted_tiles):
                if boundary_check(tile):
                    continue

                next_tile = get_next_tile(tile)
                if not next_tile:
                    tile.move(delta)
                elif (
                    tile.value == next_tile.value
                    and tile not in blocks
                    and next_tile not in blocks
                ):
                    if merge_check(tile, next_tile):
                        tile.move(delta)
                    else:
                        next_tile.value *= 2
                        self.grade += next_tile.value
                        sorted_tiles.pop(i)
                        blocks.add(next_tile)
                elif move_check(tile, next_tile):
                    tile.move(delta)
                else:
                    continue

                tile.set_pos(ceil)
                updated = True

            self.update_tiles(sorted_tiles)

        return self.end_move()

    def end_move(self):
        if len(self.tiles) == 16:
            return "lost"

        row, col = self.get_random_pos()
        # 修改这里，只生成值为2的新Tile
        self.tiles[f"{row}{col}"] = Tile(2, row, col, self.RECT_WIDTH, self.RECT_HEIGHT)
        return "continue"

    def update_tiles(self, sorted_tiles):
        self.tiles.clear()
        for tile in sorted_tiles:
            self.tiles[f"{tile.row}{tile.col}"] = tile

        self.draw()

    def generate_tiles(self):
        for _ in range(2):
            row, col = self.get_random_pos()
            self.tiles[f"{row}{col}"] = Tile(2, row, col, self.RECT_WIDTH, self.RECT_HEIGHT)

    def generate_button(self):
        BUTTON_FONT = pygame.font.SysFont(None, 80)
        button_text_surface = BUTTON_FONT.render("advice", True, (255, 255, 255))
        button_text_rect = button_text_surface.get_rect(center=pygame.Rect(800, 0, 250, 150).center)
        # Draw the button onto the given window
        button_rect = pygame.Rect(800, 0, 250, 150)
        button_fill_color = (143, 122, 102)
        pygame.draw.rect(self.WINDOW, button_fill_color, button_rect)
        self.WINDOW.blit(button_text_surface, button_text_rect)

        BUTTON_FONT = pygame.font.SysFont(None, 80)
        button_text_surface = BUTTON_FONT.render("AI", True, (255, 255, 255))
        button_text_rect = button_text_surface.get_rect(center=pygame.Rect(800, 300, 250, 150).center)
        # Draw the button onto the given window
        button_rect = pygame.Rect(800, 300, 250, 150)
        button_fill_color = (143, 122, 102)
        pygame.draw.rect(self.WINDOW, button_fill_color, button_rect)
        self.WINDOW.blit(button_text_surface, button_text_rect)

    def generate_advice(self):
        self.advice_.update(self.matrix_to_np())
        direction = self.advice_.predict_next_move()
        # 将方向转换为字符串
        direction_str = 'tips:' + str(direction)

        # 渲染文本
        text_surface = font.render(direction_str, True, (255, 255, 255))  # 文本颜色为白色

        # 获取文本的尺寸
        text_rect = text_surface.get_rect()

        # 设置文本的位置，使其在矩形下方居中
        text_rect.midtop = (800 + 250 / 2, 200)  # 矩形中心的x坐标，矩形底部的y坐标

        # 绘制文本到屏幕上
        self.WINDOW.blit(text_surface, text_rect)

    def generate_AI_status(self):
        string_ = self.ai_active
        # 将方向转换为字符串
        if self.ai_active:
            direction = 'active'
        else:
            direction = 'inactive'
        direction_str = 'AI:' + str(direction)

        # 渲染文本
        text_surface = font1.render(direction_str, True, (255, 255, 255))  # 文本颜色为白色

        # 获取文本的尺寸
        text_rect = text_surface.get_rect()

        # 设置文本的位置，使其在矩形下方居中
        text_rect.midtop = (800 + 250 / 2, 500)  # 矩形中心的x坐标，矩形底部的y坐标
        grade_text = 'grade:'  + str(self.grade)
        grade_text_surface = font1.render(grade_text, True, (255, 255, 255))
        grade_rect = grade_text_surface.get_rect()
        grade_rect.midtop = (800 + 250 / 2, 550)
        # 绘制文本到屏幕上
        self.WINDOW.blit(text_surface, text_rect)
        self.WINDOW.blit(grade_text_surface, grade_rect)


    def matrix_to_np(self):
        matrix = np.zeros((self.ROWS, self.COLS), dtype=int)
        for key, tile in self.tiles.items():
            row, col = int(key[0]), int(key[1])
            matrix[row][col] = tile.value
        return matrix

    def direction_to_english(self,direction):
        directions = ['up', 'down', 'left', 'right']
        if 0 <= direction <= 3:
            return directions[direction]
        else:
            return None  # 或者抛出异常

    def main(self):
        run = True

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

                if event.type == pygame.KEYDOWN:
                    if self.ai_active:
                        # 如果AI处于活动状态，则忽略键盘输入
                        continue

                    if event.key == pygame.K_LEFT:
                        self.move_tiles("left")
                    elif event.key == pygame.K_RIGHT:
                        self.move_tiles("right")
                    elif event.key == pygame.K_UP:
                        self.move_tiles("up")
                    elif event.key == pygame.K_DOWN:
                        self.move_tiles("down")

                # Handle mouse clicks
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()

                    if pygame.Rect(800, 0, 250, 150).collidepoint(mouse_pos):
                       # print(self.matrix_to_np())
                        self.isGerenateAdvice = True

                    if pygame.Rect(800, 300, 250, 150).collidepoint(mouse_pos):
                        self.ai_active = not self.ai_active  # 切换AI活动状态
                       # print("AI" + (" activated" if self.ai_active else " deactivated"))

            try:
                if self.ai_active:
                    # AI自动操作：根据当前游戏状态，获取AI的最佳移动方向
                    self.ai = AI(mat(self.matrix_to_np(),self.smooth_weight, self.mono_weight, self.empty_weight, self.max_weight))
                    best_move = self.ai.getBest()
                    # 假设您的AI类有一个getBestMove方法返回最佳移动方向
                    # print('移动方向:',best_move['move'])
                    if best_move['move'] == -1:
                        print('?')
                    self.move_tiles(self.direction_to_english(best_move['move']))
            except Exception as e:
                # 这里可以记录错误日志或者进行其他错误处理
               # print(f"An error occurred: {e}")
                self.ai_active = False
                return self.grade

            self.draw()

        pygame.quit()
