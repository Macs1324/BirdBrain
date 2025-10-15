import os
import random

import pygame

import quickbrain as qb

# A playground to test out how the best bird does, without having to train it
# or see the other birds

pygame.init()
SIZE = [400, 708]
FONT = pygame.font.SysFont("arialrounded", 50)
FPS = 30


class Bird:
    def __init__(self):
        self.brain = qb.load("auto_bestbrain.qb")
        self.fitness = 0
        self.x = 50
        self.y = 350
        self.jump = 0
        self.jump_speed = 10
        self.gravity = 10
        self.dead = False
        self.sprite = 0
        self.bird_sprites = [
            pygame.image.load("images/1.png").convert_alpha(),
            pygame.image.load("images/2.png").convert_alpha(),
            pygame.image.load("images/dead.png").convert_alpha(),
        ]
        # self.img_rect =

    def move(self):
        if self.dead:  # dead bird
            self.sprite = 2  # change to dead.png
            # keeps falling until it hits the ground
            if self.y < SIZE[1] - 30:
                self.y += self.gravity
        elif self.y > 0:
            # handling movement while jumping
            if self.jump:
                self.sprite = 1  # change to 2.png
                self.jump_speed -= 1
                self.y -= self.jump_speed
            else:
                # regular falling (increased gravity)
                self.gravity += 0.2
                self.y += self.gravity
        else:
            # in-case where the bird reaches the top
            # of the screen
            self.jump = 0
            self.y += 3

    def bottom_check(self):
        # bird hits the bottom = DEAD
        if self.y >= SIZE[1] - 30:
            self.dead = True

    def get_rect(self):
        # updated bird image rectangle
        img_rect = self.bird_sprites[self.sprite].get_rect()
        img_rect[0] = self.x
        img_rect[1] = self.y
        return img_rect


class Pillar:
    def __init__(self, pos):
        # pos == True is top , pos == False is bottom
        self.pos = pos
        self.img = self.get_image()

    def get_rect(self):
        # returns the pillar image rect
        return self.img.get_rect()

    def get_image(self):
        if self.pos:  # image for the top pillar
            return pygame.image.load("images/top.png").convert_alpha()
        else:  # image for the bottom pillar
            return pygame.image.load("images/bottom.png").convert_alpha()


class Options:
    def __init__(self):
        self.score_img = pygame.image.load(
            "images/score.png"
        ).convert_alpha()  # score board image
        self.play_img = pygame.image.load(
            "images/play.png"
        ).convert_alpha()  # play button image
        self.play_rect = self.play_img.get_rect()
        self.score_rect = self.score_img.get_rect()
        self.align_position()
        self.score = 0
        self.font = FONT

    def align_position(self):
        # aligns the "menu" in certain positions
        self.play_rect.center = (200, 330)
        self.score_rect.center = (200, 220)

    def inc(self):
        # score increased by 1
        self.score += 1


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SIZE[0], SIZE[1]))
        pygame.display.set_caption("Flappy Bird")
        self.background = pygame.image.load(
            "images/background.png"
        ).convert()  # background image
        self.pillar_x = 400
        self.offset = 0
        self.top_p = Pillar(1)  # top pillar
        self.bot_p = Pillar(0)  # bottom pillar
        self.pillar_gap = 135  # gap between pillars, (can be randomised as well)
        self.bird = Bird()  # bird object
        self.score_board = Options()
        self.passed = False  # allows to keep track of the score

    def pillar_move(self):
        # handling pillar movement in the background
        if self.pillar_x < -100:
            self.offset = random.randrange(-120, 120)
            self.passed = False
            self.pillar_x = 400
        self.pillar_x -= 5

    def get_gap_coords(self):
        gap_x = self.get_pillar_rect(self.top_p).x
        gap_y = (
            self.get_pillar_rect(self.top_p).bottom
            + self.get_pillar_rect(self.bot_p).top
        )
        gap_y /= 2
        gap_x = int(gap_x)
        gap_y = int(gap_y)

        return (gap_x, gap_y)

    def run(self):
        clock = pygame.time.Clock()
        done = True
        while done:
            gap = self.get_gap_coords()
            q = qb.qm.Matrix(data=[[self.bird.y, gap[0], gap[1]]]).transpose()
            self.bird.decision = self.bird.brain.feed_forward(q)[0][0]
            if self.bird.decision >= 0.5:
                self.bird.jump = 17
                self.bird.gravity = 5
                self.bird.jump_speed = 10
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # bird jumps
                        self.bird.jump = 17
                        self.bird.gravity = 5
                        self.bird.jump_speed = 10
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # clicking on the play button (game reset)
                    if self.bird.dead and self.score_board.play_rect.collidepoint(
                        event.pos
                    ):
                        self.bird.dead = False
                        self.reset()

            self.screen.blit(self.background, (0, 0))
            self.screen.blit(
                self.top_p.img, (self.pillar_x, 0 - self.pillar_gap - self.offset)
            )
            self.screen.blit(
                self.bot_p.img, (self.pillar_x, 360 + self.pillar_gap - self.offset)
            )
            self.screen.blit(
                self.bird.bird_sprites[self.bird.sprite], (self.bird.x, self.bird.y)
            )
            self.pillar_move()
            self.bird.move()
            self.bird.bottom_check()
            if not self.bird.dead:
                self.collision()
                self.show_score()
            else:
                self.game_over()
            self.draw_neural_network()  # Draw neural network visualization
            pygame.display.flip()

    def get_pillar_rect(self, pillar):
        # returns current pillar rectangle on display
        rect = pillar.get_image().get_rect()
        rect[0] = self.pillar_x
        if pillar.pos:
            # current rect y position for top pillar
            rect[1] = 0 - self.pillar_gap - self.offset
        else:
            # current rect y position for bottom pillar
            rect[1] = 360 + self.pillar_gap - self.offset
        return rect

    def collision(self):
        top_rect = self.get_pillar_rect(self.top_p)
        bot_rect = self.get_pillar_rect(self.bot_p)
        # collision check bird <> pillars
        if top_rect.colliderect(self.bird.get_rect()) or bot_rect.colliderect(
            self.bird.get_rect()
        ):
            # print(self.bird.bird_sprites[self.bird.sprite].get_rect())
            self.bird.dead = True
        # if bird passed the pillars
        elif not self.passed and top_rect.right < self.bird.x:
            self.score_board.inc()
            self.passed = True

    def reset(self):
        # game values reset
        self.score_board.score = 0
        self.bird = Bird()
        self.top_p = Pillar(1)
        self.bot_p = Pillar(0)
        self.pillar_x = 400
        self.bird.gravity = 10

    def show_score(self):
        # score font
        score_font = FONT.render(
            "{}".format(self.score_board.score), True, (255, 80, 80)
        )
        # score font rectangle
        font_rect = score_font.get_rect()
        font_rect.center = (200, 50)
        self.screen.blit(score_font, font_rect)  # show score board font

    def game_over(self):
        # score font
        score_font = FONT.render(
            "{}".format(self.score_board.score), True, (255, 80, 80)
        )
        # score font rectangle
        font_rect = score_font.get_rect()
        score_rect = self.score_board.score_rect
        play_rect = self.score_board.play_rect  # play button rectangle
        font_rect.center = (200, 230)
        self.screen.blit(self.score_board.play_img, play_rect)  # show play button
        self.screen.blit(
            self.score_board.score_img, score_rect
        )  # show score board image
        self.screen.blit(score_font, font_rect)  # show score font

    def draw_neural_network(self):
        # Visualize neural network activations in bottom right corner
        viz_width = 120
        viz_height = 150
        viz_x = SIZE[0] - viz_width - 10
        viz_y = SIZE[1] - viz_height - 10

        # Semi-transparent background
        surface = pygame.Surface((viz_width, viz_height))
        surface.set_alpha(60)
        surface.fill((30, 30, 30))
        self.screen.blit(surface, (viz_x, viz_y))

        # Get activations from the brain (Custom_DFF stores layers in self.layers)
        input_layer = self.bird.brain.layers[0].matrix  # first layer
        hidden_layers = [layer.matrix for layer in self.bird.brain.layers[1:-1]]  # middle layers
        output_layer = self.bird.brain.layers[-1].matrix  # last layer

        # Neuron visualization parameters
        neuron_radius = 5
        layer_spacing = 35
        viz_center_y = viz_y + viz_height // 2

        # Calculate positions for each layer
        input_x = viz_x + 20
        hidden_x = viz_x + 20 + layer_spacing
        output_x = viz_x + 20 + 2 * layer_spacing

        # Calculate vertical positions for input layer (centered)
        input_positions = []
        input_height = len(input_layer) * 20
        input_y_start = viz_center_y - input_height // 2
        for i in range(len(input_layer)):
            y_pos = input_y_start + i * 20
            input_positions.append((input_x, y_pos))

        # Calculate vertical positions for hidden layer (centered)
        all_hidden_neurons = []
        for hidden_layer in hidden_layers:
            all_hidden_neurons.extend(hidden_layer)

        hidden_positions = []
        if all_hidden_neurons:
            hidden_height = len(all_hidden_neurons) * 15
            hidden_y_start = viz_center_y - hidden_height // 2
            for i in range(len(all_hidden_neurons)):
                y_pos = hidden_y_start + i * 15
                hidden_positions.append((hidden_x, y_pos))

        # Calculate vertical position for output layer (centered)
        output_positions = [(output_x, viz_center_y)]

        # Draw connections (weights) between layers
        # Input to hidden
        for input_pos in input_positions:
            for hidden_pos in hidden_positions:
                pygame.draw.line(self.screen, (80, 80, 80), input_pos, hidden_pos, 1)

        # Hidden to output
        for hidden_pos in hidden_positions:
            for output_pos in output_positions:
                pygame.draw.line(self.screen, (80, 80, 80), hidden_pos, output_pos, 1)

        # Font for activation values
        tiny_font = pygame.font.SysFont("arialrounded", 8)

        # Draw input layer neurons
        for i, (neuron, pos) in enumerate(zip(input_layer, input_positions)):
            activation = neuron[0]
            # Grayscale: higher activation = brighter
            color_val = min(255, max(0, int(activation * 255)))
            color = (color_val, color_val, color_val)
            pygame.draw.circle(self.screen, color, pos, neuron_radius)
            pygame.draw.circle(self.screen, (150, 150, 150), pos, neuron_radius, 1)  # border
            # Draw activation value
            act_text = tiny_font.render(f"{activation:.2f}", True, (255, 255, 255))
            text_rect = act_text.get_rect(center=(pos[0], pos[1] - 10))
            self.screen.blit(act_text, text_rect)

        # Draw hidden layer neurons
        if all_hidden_neurons:
            for i, (neuron, pos) in enumerate(zip(all_hidden_neurons, hidden_positions)):
                activation = neuron[0]
                color_val = min(255, max(0, int(activation * 255)))
                color = (color_val, color_val, color_val)
                pygame.draw.circle(self.screen, color, pos, neuron_radius)
                pygame.draw.circle(self.screen, (150, 150, 150), pos, neuron_radius, 1)  # border
                # Draw activation value
                act_text = tiny_font.render(f"{activation:.2f}", True, (255, 255, 255))
                text_rect = act_text.get_rect(center=(pos[0], pos[1] - 10))
                self.screen.blit(act_text, text_rect)

        # Draw output layer neurons
        for neuron, pos in zip(output_layer, output_positions):
            activation = neuron[0]
            color_val = min(255, max(0, int(activation * 255)))
            color = (color_val, color_val, color_val)
            pygame.draw.circle(self.screen, color, pos, neuron_radius)
            pygame.draw.circle(self.screen, (150, 150, 150), pos, neuron_radius, 1)  # border
            # Draw activation value
            act_text = tiny_font.render(f"{activation:.2f}", True, (255, 255, 255))
            text_rect = act_text.get_rect(center=(pos[0], pos[1] - 10))
            self.screen.blit(act_text, text_rect)

        # Draw labels
        small_font = pygame.font.SysFont("arialrounded", 12)
        title_text = small_font.render("Brain", True, (200, 200, 200))
        self.screen.blit(title_text, (viz_x + 5, viz_y + 5))


# os.chdir(os.path.dirname(__file__))
if __name__ == "__main__":
    game = Game()
    game.run()
