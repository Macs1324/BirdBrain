import pygame
import random
import os
import quickbrain as qb
import QuickMaths as qm


#Training hall for birds:
#   Runs a Genetic algorithm (with tunable parameters)
#   to train the birds
#WARNING: Running a bunch of NNs built on python lists is horrible, performance might suffer!


POPULATION_SIZE = 100
MUTATION_RATE = 10

BIRD_TRESHOLD = 0.5

PILLAR_SPEED = 10
PILLAR_MAX_SPEED = 30

NORMAL_GRAVITY = 10
SUPER_GRAVITY = 30

MIN_GAP = 120
MAX_GAP = 200

def search(_list, element):
    for i in range(len(_list)):
        if _list[i] == element:
            return i
            break
    return 0

def get_genotype(brain):
    return brain.weights, brain.biases

def set_genotype(brain, w, b):
    brain.weights = w
    brain.biases = b

def mutate(brain, rate=0.1):
    for i in range(len(brain.weights)):
        mutation = qm.random_matrix(brain.weights[i].rows, brain.weights[i].cols) * rate
        brain.weights[i] += mutation
    for i in range(len(brain.biases)):
        mutation = qm.random_matrix(brain.biases[i].rows, brain.biases[i].cols) * rate
        brain.biases[i] += mutation

def crossover(w1, b1, w2, b2):
    r_w = [None for i in range(len(w1))]
    r_b = [None for i in range(len(b1))]
    for i in range(len(w1)):
        decision = random.randint(0, 100)
        if decision < 50:
            r_w[i] = w1[i]
        else:
            r_w[i] = w2[i]
    for i in range(len(b1)):
        decision = random.randint(0, 100)
        if decision < 50:
            r_b[i] = b2[i]
        else:
            r_b[i] = b1[i]

    return r_w, r_b




pygame.init()
SIZE = [400, 708]
FONT = pygame.font.SysFont('arialrounded', 50)


class Bird:
    def __init__(self):
        self.fitness = 0
        self.decision = 0
        self.brain = qb.Custom_DFF([5, 2], qm.Activations.sigmoid)
        self.x = 50
        self.y = 350
        self.jump = 0
        self.jump_speed = 10
        self.gravity = 10
        self.dead = False
        self.sprite = 0
        self.bird_sprites = [pygame.image.load("images/1.png").convert_alpha(),
                             pygame.image.load("images/2.png").convert_alpha(),
                             pygame.image.load("images/dead.png").convert_alpha()]

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
        if self.y >= SIZE[1] - 30 or self.y < 0:
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
        self.score_img = pygame.image.load("images/score.png").convert_alpha()  # score board image
        self.play_img = pygame.image.load("images/play.png").convert_alpha()  # play button image
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
        self.pillar_speed = PILLAR_SPEED
        self.bestbrain = None
        self.screen = pygame.display.set_mode((SIZE[0], SIZE[1]))
        pygame.display.set_caption("Flappy Bird")
        self.background = pygame.image.load("images/background.png").convert()  # background image
        self.pillar_x = 400
        self.offset = 0
        self.top_p = Pillar(1)  # top pillar
        self.bot_p = Pillar(0)  # bottom pillar
        self.pillar_gap = random.randint(MIN_GAP, MAX_GAP)  # gap between pillars, (can be randomised as well)
        self.birds = []  # bird object
        for i in range(POPULATION_SIZE):
            self.birds.append(Bird())
        self.score_board = Options()
        self.passed = False  # allows to keep track of the score

    def all_dead(self):
        for bird in self.birds:
            if not bird.dead:
                return False
        return True
    def pillar_move(self):
        # handling pillar movement in the background
        if self.pillar_x < -100:
            self.offset = random.randrange(-120, 120)
            self.passed = False
            self.pillar_x = 400
            self.pillar_gap = random.randint(MIN_GAP, MAX_GAP)
        if self.all_dead():
            self.pillar_gap = random.randint(MIN_GAP, MAX_GAP)
        if self.pillar_speed < PILLAR_MAX_SPEED:
            self.pillar_speed += 0.001
        self.pillar_x -= self.pillar_speed

    def get_gap_coords(self):
        gap_x = self.get_pillar_rect(self.top_p).x
        gap_y = self.get_pillar_rect(self.top_p).bottom + self.get_pillar_rect(self.bot_p).top
        gap_y /= 2
        gap_x = int(gap_x)
        gap_y = int(gap_y)

        return (gap_x, gap_y)
    def run(self):
        clock = pygame.time.Clock()
        done = True
        while done:
            #self.background.fill((0,0,0))
            pygame.display.update()
            #pygame.draw.circle(self.background, 3, self.get_gap_coords(), 10)
            #print(self.birds[0].fitness)
            for bird in self.birds:
                if not bird.dead:
                    #print(bird.decision)
                    bird.fitness += 1
                    gap = self.get_gap_coords()
                    distance = qm.math.sqrt((gap[0] - bird.x) **2 + (gap[1] - bird.y) ** 2)
                    bird.fitness += 100 / distance
                bird.decision = bird.brain.feed_forward(qm.Matrix(data=[[self.pillar_speed, self.pillar_gap, bird.y, gap[0], gap[1]]]).transpose())
                if bird.decision[0][0] >= BIRD_TRESHOLD:
                    bird.jump = 17
                    bird.gravity = 5
                    bird.jump_speed = 10
                if bird.decision[1][0] >= BIRD_TRESHOLD:
                    bird.gravity = SUPER_GRAVITY
                else:
                    bird.gravity = NORMAL_GRAVITY
            clock.tick(60)
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_s]:
                qb.save(self.bestbrain, "bestbrain.qb")
                print("saving...")
            if pressed[pygame.K_l]:
                print("Loading...")
                self.birds = [Bird() for i in range(2)]
                self.birds[-1].brain = qb.load("bestbrain.qb")
                self.pillar_x = 400
            if pressed[pygame.K_r]:
                self.reset()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()


            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.top_p.img, (self.pillar_x, 0 - self.pillar_gap - self.offset))
            self.screen.blit(self.bot_p.img, (self.pillar_x, 360 + self.pillar_gap - self.offset))
            self.pillar_move()
            for bird in self.birds:
                self.screen.blit(bird.bird_sprites[bird.sprite], (bird.x, bird.y))
                bird.move()
                bird.bottom_check()
                if not self.all_dead():
                    self.collision()
                    self.show_score()
                else:
                    self.reset()
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
        for bird in self.birds:
            if top_rect.colliderect(bird.get_rect()) or bot_rect.colliderect(bird.get_rect()):
            # print(self.bird.bird_sprites[self.bird.sprite].get_rect())
                bird.dead = True
        # if bird passed the pillars
            elif not self.passed and top_rect.right < bird.x:
                self.score_board.inc()
                self.passed = True
                bird.fitness += 1000

    def reset(self):
        #print("reset")
        # game values reset
        self.score_board.score = 0
        self.top_p = Pillar(1)
        self.bot_p = Pillar(0)
        self.pillar_x = 400
        for bird in self.birds:
            bird.gravity = 10

        scores = []
        for bird in self.birds:
            scores.append(bird.fitness)
        top_score_i = search(scores, max(scores))

        del scores[top_score_i]

        mama = self.birds[top_score_i]
        self.bestbrain = mama.brain
        qb.save(self.bestbrain, "auto_bestbrain.qb")
        mama.dead = False
        top_score_i = search(scores, max(scores))
        papa = self.birds[top_score_i]

        self.birds = [Bird()]
        set_genotype(self.birds[0].brain, mama.brain.weights, mama.brain.biases)
        for i in range(POPULATION_SIZE):
            baby = Bird()
            w1, b1 = get_genotype(papa.brain)
            w2, b2 = get_genotype(mama.brain)

            genes_w, genes_b = crossover(w1, b1, w2, b2)
            child = Bird()
            set_genotype(child.brain, genes_w, genes_b)
            mutate(child.brain, MUTATION_RATE)
            self.birds.append(child)
        self.birds[0].brain = self.bestbrain
        self.pillar_speed = 10

    def show_score(self):
        # score font
        score_font = FONT.render("{}".format(self.score_board.score),
                                               True, (255, 80, 80))
        # score font rectangle
        font_rect = score_font.get_rect()
        font_rect.center = (200, 50)
        self.screen.blit(score_font, font_rect)  # show score board font

    def game_over(self):
        # score font
        score_font = FONT.render("{}".format(self.score_board.score),
                                     True, (255, 80, 80))
        # score font rectangle
        font_rect = score_font.get_rect()
        score_rect = self.score_board.score_rect
        play_rect = self.score_board.play_rect  # play button rectangle
        font_rect.center = (200, 230)
        self.screen.blit(self.score_board.play_img, play_rect)  # show play button
        self.screen.blit(self.score_board.score_img, score_rect)  # show score board image
        self.screen.blit(score_font, font_rect)  # show score font


#os.chdir(os.path.dirname(__file__))
if __name__ == "__main__":
    game = Game()
    game.run()
