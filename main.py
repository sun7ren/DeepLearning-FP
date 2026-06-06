from __future__ import annotations

import os
import sys
import warnings
from itertools import batched
from pathlib import Path
from random import gauss, randint, random, seed, uniform
import matplotlib.pyplot as plt
import numpy as np
import copy

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pygame
import torch
from deap import (
    algorithms as deap_algorithms,
    base as deap_base,
    creator as deap_creator,
    tools as deap_tools,
)
from torch import nn

from game import FPS, FlappyBirdGame
from game import (
    HEIGHT as SCREEN_HEIGHT,
    WIDTH as SCREEN_WIDTH,
)

# Types
type GameState2 = tuple[float, float]  # dx, dy
type GameState3 = tuple[float, float, float]  # dx, dy, y

# Static parameters
type GameState = GameState3

OPTIMIZER = torch.optim.Adam  # Optimized used during the reinforcment learning
EPOCHS = 100  # Number of epochs on which each model is trained
MAX_SCORE = 10  # Stop each run at this score (to avoid infinite runs)

POPULATION_SIZE = 50
GENERATIONS = 50
MUTATION_PROBABILITY = 0.2

RANDOM_SEED = 42

# Dynamic parameters
N_INPUTS = int(str(GameState.__value__)[len("GameState") :])

ACTIVATIONS = (
    nn.ReLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.LeakyReLU,
    nn.ELU,
)

MAX_HIDDEN_LAYERS = 10
MIN_LAYER_SIZE = 1
MAX_LAYER_SIZE = 10
MIN_LEARNING_RATE = 1e-4
MAX_LEARNING_RATE = 1e-1

# [layer_size | activation idx] * MAX_HIDDEN_LAYERS | nb hidden layers | learning rate
type Genome = list[*([int, int] * MAX_HIDDEN_LAYERS), int, float]


# Genetics
def create_random_genome() -> Genome:
    genome = []
    for _ in range(MAX_HIDDEN_LAYERS):
        genome.append(randint(MIN_LAYER_SIZE, MAX_LAYER_SIZE))
        genome.append(randint(0, len(ACTIVATIONS) - 1))
    genome.append(randint(0, MAX_HIDDEN_LAYERS))
    genome.append(
        10 ** uniform(np.log10(MIN_LEARNING_RATE), np.log10(MAX_LEARNING_RATE))
    )
    return genome


def build_mlp(genome: Genome) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    layers_data = genome[:-2]
    last_layer = genome[-2]
    learning_rate = genome[-1]

    layers = []
    in_dim = N_INPUTS
    for layer_size, activation_idx in batched(layers_data[: last_layer * 2], n=2):
        layers.append(nn.Linear(in_dim, layer_size))
        layers.append(ACTIVATIONS[activation_idx]())
        in_dim = layer_size
    layers.append(nn.Linear(in_dim, 1))
    model = nn.Sequential(*layers)
    optimizer = OPTIMIZER(model.parameters(), lr=learning_rate)
    return model, optimizer


_best_model_cache = None
_best_fitness_cache = -1


def evaluate_genome(genome: Genome) -> tuple[float]:
    global _best_fitness_cache, _best_model_cache

    model, optimizer = build_mlp(genome)

    # Training
    model.train()
    for _ in range(EPOCHS):
        game = FlappyBirdGame(None, None, None, render=False)

        log_probs = []
        rewards = []
        old_score = game.score

        while True:
            y, dx, dy = game.get_state()

            if N_INPUTS == 2:
                game_state = dx, dy
            elif N_INPUTS == 3:
                game_state = dx, dy, y
            else:
                raise NotImplementedError(
                    f"missing game state implementation for N_INPUTS={N_INPUTS}"
                )
            game_state = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0)

            logits = model(game_state)
            probs = torch.sigmoid(logits)
            dist = torch.distributions.Bernoulli(probs)
            action = dist.sample()
            action_val = action.item()
            log_probs.append(dist.log_prob(action))
            game.step(int(action_val))

            if game.done:
                rewards.append(-100)
                break

            if game.score > old_score:
                rewards.append(100)
            else:
                rewards.append(-0.01)

                
            old_score = game.score

        gamma = 0.99
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

        loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()

    game = FlappyBirdGame(None, None, None, render=False)
    old_score = game.score
    fitness = 0

    while True:
        y, dx, dy = game.get_state()

        if N_INPUTS == 2:
            game_state = dx, dy
        elif N_INPUTS == 3:
            game_state = dx, dy, y
        else:
            raise NotImplementedError(
                f"missing game state implementation for N_INPUTS={N_INPUTS}"
            )
        game_state = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model(game_state)
            should_jump = (logits > 0).item()

        game.step(1 if should_jump else 0)

        if game.done:
            break
        elif game.score >= MAX_SCORE:
            fitness += 100
        elif game.score > old_score:
            fitness += 10
        else:
            fitness += 1
        

        old_score = game.score

    if fitness > _best_fitness_cache:
        _best_fitness_cache = fitness
        _best_model_cache = model

    return (fitness,)

def smooth(data, window=5):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def mutate(genome: Genome, ind_pb: float) -> Genome:
    for i in range(MAX_HIDDEN_LAYERS):
        if random() < ind_pb:
            genome[2 * i] = randint(MIN_LAYER_SIZE, MAX_LAYER_SIZE)
        if random() < ind_pb:
            genome[2 * i + 1] = randint(0, len(ACTIVATIONS) - 1)

    if random() < ind_pb:
        genome[-2] = randint(1, MAX_HIDDEN_LAYERS)

    if random() < ind_pb:
        log_lr = np.log10(max(genome[-1], 1e-8))
        genome[-1] = 10 ** np.clip(
            log_lr + gauss(0, 0.5),
            np.log10(MIN_LEARNING_RATE),
            np.log10(MAX_LEARNING_RATE),
        )

    return (genome,)


# Mains
def train(save_path: Path):
    deap_creator.create("FitnessMax", deap_base.Fitness, weights=(1.0,))
    deap_creator.create("Individual", list, fitness=deap_creator.FitnessMax)

    deap_toolbox = deap_base.Toolbox()
    deap_toolbox.register(
        "individual",
        deap_tools.initIterate,
        deap_creator.Individual,
        create_random_genome,
    )
    deap_toolbox.register(
        "population", deap_tools.initRepeat, list, deap_toolbox.individual
    )
    deap_toolbox.register("mate", deap_tools.cxTwoPoint)
    deap_toolbox.register("mutate", mutate, ind_pb=MUTATION_PROBABILITY)
    deap_toolbox.register("select", deap_tools.selTournament, tournsize=3)
    deap_toolbox.register("evaluate", evaluate_genome)

    population = deap_toolbox.population(n=POPULATION_SIZE)
    best_scores = []
    avg_scores = []

    stats = deap_tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    hof = deap_tools.HallOfFame(1)

    population, logbook = deap_algorithms.eaSimple(
        population,
        deap_toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    best_scores = logbook.select("max")
    avg_scores = logbook.select("avg")

    model_filepath = save_path.with_suffix(".pytorch")
    model_filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "genome": hof[0],
            "fitness": hof[0].fitness.values[0],
            "state_dict": _best_model_cache.state_dict(),
            "n_inputs": N_INPUTS,
        },
        model_filepath,
    )

    plt.figure(figsize=(10,5))

    plt.plot(smooth(best_scores), label="Best Fitness (smoothed)")
    plt.plot(smooth(avg_scores), label="Avg Fitness (smoothed)")

    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title("Flappy Bird Learning Curve")
    plt.legend()
    plt.grid()

    plt.show()


def display(load_path: Path):
    model_filepath = load_path.with_suffix(".pytorch")
    data = torch.load(model_filepath, map_location="cpu", weights_only=False)

    best_genome = data["genome"]
    n_inputs = data["n_inputs"]

    model, _ = build_mlp(best_genome)
    model.load_state_dict(data["state_dict"])
    model.eval()

    print((f"displaying result of genome:\n{best_genome}\n=> fitness={n_inputs}"))

    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Phoenix Navigation")

    font = pygame.font.SysFont(None, 40)

    background_img = pygame.image.load("assets/background.png").convert()
    background_img = pygame.transform.scale(
        background_img, (SCREEN_WIDTH, SCREEN_HEIGHT)
    )

    bird_img = pygame.image.load("assets/phoenix.png").convert_alpha()
    bird_img = pygame.transform.smoothscale(bird_img, (90, 78))

    pipe_img = pygame.image.load("assets/bars.png").convert_alpha()

    game = FlappyBirdGame(background_img, bird_img, pipe_img, render=True)

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif game.done and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game.reset()

        if not game.done:
            y, dx, dy = game.get_state()

            if n_inputs == 2:
                game_state = dx, dy
            elif n_inputs == 3:
                game_state = dx, dy, y
            else:
                raise NotImplementedError(
                    f"missing game state implementation for N_INPUTS={N_INPUTS}"
                )
            game_state = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                logits = model(game_state)
                should_jump = (logits > 0).item()

            game.step(1 if should_jump else 0)

        game.draw(screen, font)

        if game.done:
            text = font.render("GAME OVER (R to restart)", True, (255, 0, 0))
            screen.blit(text, (70, SCREEN_HEIGHT // 2))

        pygame.display.flip()


if __name__ == "__main__":
    seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    pygame.init()

    BASE_DIR = Path(__file__).parent

    if len(sys.argv) > 1 and sys.argv[1] == "--display":
        display(BASE_DIR / "saves" / "best")
    else:
        train(BASE_DIR / "saves" / "best")
