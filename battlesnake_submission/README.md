# Battlesnake Python Starter Project

An official Battlesnake template written in Python. Get started at [play.battlesnake.com](https://play.battlesnake.com).

![Battlesnake Logo](https://media.battlesnake.com/social/StarterSnakeGitHubRepos_Python.png)

This project is a great starting point for anyone wanting to program their first Battlesnake in Python. It can be run locally or easily deployed to a cloud provider of your choosing. See the [Battlesnake API Docs](https://docs.battlesnake.com/api) for more detail. 

[![Run on Replit](https://repl.it/badge/github/BattlesnakeOfficial/starter-snake-python)](https://replit.com/@Battlesnake/starter-snake-python)

## Technologies Used

This project uses [Python 3](https://www.python.org/) and [Flask](https://flask.palletsprojects.com/). It also comes with an optional [Dockerfile](https://docs.docker.com/engine/reference/builder/) to help with deployment.

## Install

Install dependencies using pip:

```sh
pip install -r requirements.txt
```

## Run Each Agent

```sh
python random_main.py
python heuristic_main.py
python mcts_vanilla_main.py
python mcts_main.py
```

These commands run the following agents:

- `python random_main.py`: Random
- `python heuristic_main.py`: Heuristic
- `python mcts_vanilla_main.py`: MCTS-Vanilla
- `python mcts_main.py`: MCTS-All
