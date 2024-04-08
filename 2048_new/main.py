import numpy as np
import pygame
import random
import math

pygame.init()

FPS = 60

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
    def __init__(self):
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
        for tile in self.tiles.values():
            tile.draw(self.WINDOW)

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
        self.tiles[f"{row}{col}"] = Tile(random.choice([2, 4]), row, col, self.RECT_WIDTH, self.RECT_HEIGHT)
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

    def matrix_to_np(self):
        matrix = np.zeros((self.ROWS, self.COLS), dtype=int)
        for key, tile in self.tiles.items():
            row, col = int(key[0]), int(key[1])
            matrix[row][col] = tile.value
        return matrix

    def main(self):
        run = True

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move_tiles("left")
                    if event.key == pygame.K_RIGHT:
                        self.move_tiles("right")
                    if event.key == pygame.K_UP:
                        self.move_tiles("up")
                    if event.key == pygame.K_DOWN:
                        self.move_tiles("down")

                # Handle mouse clicks
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if pygame.Rect(800, 0, 250, 150).collidepoint(mouse_pos):
                        print(self.matrix_to_np())
                    if pygame.Rect(800, 300, 250, 150).collidepoint(mouse_pos):
                        print("Button 'AI' clicked!")

            self.draw()

        pygame.quit()

if __name__ == "__main__":
    game = Game2048()
    game.main()
