import pygame
import random
import sys

WIDTH = 500
HEIGHT = 700
GRAVITY = 0.5
JUMP_STRENGTH = -8
PIPE_WIDTH = 80
PIPE_GAP = 260
PIPE_SPEED = 4

FPS = 60


class Bird:
    def __init__(self):
        self.x = 120
        self.y = HEIGHT // 2
        self.width = 70
        self.height = 60
        self.velocity = 0

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def flap(self):
        self.velocity = JUMP_STRENGTH

    def draw(self, screen, image):
        screen.blit(
            image,
            (self.x - self.width // 2, self.y - self.height // 2)
        )

    def get_rect(self):
        return pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height
        )


class Pipe:
    def __init__(self):
        self.x = WIDTH
        self.gap_y = random.randint(150, HEIGHT - 150)
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self, screen, pipe_image):
        top_height = self.gap_y - PIPE_GAP // 2
        bottom_y = self.gap_y + PIPE_GAP // 2

        top_pipe = pygame.transform.flip(pipe_image, False, True)
        top_pipe = pygame.transform.scale(top_pipe, (PIPE_WIDTH, top_height))
        screen.blit(top_pipe, (self.x, 0))

        bottom_pipe = pygame.transform.scale(
            pipe_image,
            (PIPE_WIDTH, HEIGHT - bottom_y)
        )
        screen.blit(bottom_pipe, (self.x, bottom_y))

    def collide(self, bird):
        bird_rect = bird.get_rect()
        top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.gap_y - PIPE_GAP // 2)
        bottom_rect = pygame.Rect(self.x, self.gap_y + PIPE_GAP // 2, PIPE_WIDTH, HEIGHT)

        return (
            bird_rect.colliderect(top_rect)
            or bird_rect.colliderect(bottom_rect)
        )


class FlappyBirdGame:
    def __init__(self, background_img, bird_img, pipe_img, render=True):
        self.background_img = background_img
        self.bird_img = bird_img
        self.pipe_img = pipe_img
        self.render_enabled = render
        self.reset()

    def reset(self):
        self.bird = Bird()
        self.pipes = [Pipe()]
        self.score = 0
        self.done = False

    def get_next_pipe(self):
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > self.bird.x:
                return pipe
        return self.pipes[0]

    def get_state(self):
        """
        State vector to 4 inputs:
        distance_to_ground : normalized distance from bird to bottom
        dx                 : normalized horizontal distance to next pipe
        dy                 : normalized vertical distance to pipe gap center
        velocity           : normalized vertical velocity of the bird
        """
        pipe = self.get_next_pipe()

        distance_to_ground = (HEIGHT - self.bird.y) / HEIGHT
        dx = (pipe.x - self.bird.x) / WIDTH
        dy = (pipe.gap_y - self.bird.y) / HEIGHT
        velocity = max(-1, min(1, self.bird.velocity / 10))

        return [distance_to_ground, dx, dy, velocity]

    def step(self, action):
        if action == 1:
            self.bird.flap()

        self.bird.update()
        reward = 0.1

        for pipe in self.pipes:
            pipe.update()

            if pipe.collide(self.bird):
                self.done = True
                reward = -100

            if not pipe.passed and pipe.x < self.bird.x:
                pipe.passed = True
                self.score += 1
                reward = 10

        if self.bird.y < 0 or self.bird.y > HEIGHT:
            self.done = True
            reward = -100

        if self.pipes[-1].x < WIDTH - 320:
            self.pipes.append(Pipe())

        if self.pipes[0].x < -PIPE_WIDTH:
            self.pipes.pop(0)

        return self.get_state(), reward, self.done

    def draw(self, screen, font):
        if not self.render_enabled:
            return

        screen.blit(self.background_img, (0, 0))
        self.bird.draw(screen, self.bird_img)

        for pipe in self.pipes:
            pipe.draw(screen, self.pipe_img)

        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (20, 20))


def main(render=True):
    pygame.init()
    clock = pygame.time.Clock()

    if render:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Phoenix Navigation")
        font = pygame.font.SysFont(None, 40)
        background_img = pygame.image.load("assets/background.png").convert()
        background_img = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
        bird_img = pygame.image.load("assets/phoenix.png").convert_alpha()
        bird_img = pygame.transform.smoothscale(bird_img, (90, 78))
        pipe_img = pygame.image.load("assets/bars.png").convert_alpha()
    else:
        screen, font, background_img, bird_img, pipe_img = None, None, None, None, None

    game = FlappyBirdGame(background_img, bird_img, pipe_img, render=render)

    while True:
        if render:
            clock.tick(FPS)

        action = 0
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = 1
                    if event.key == pygame.K_r and game.done:
                        game.reset()

        if not game.done:
            game.step(action)

        if render:
            game.draw(screen, font)

        if render and game.done:
            text = font.render("GAME OVER (R to restart)", True, (255, 0, 0))
            screen.blit(text, (70, HEIGHT // 2))

        if render:
            pygame.display.flip()


if __name__ == "__main__":
    main(render=True)