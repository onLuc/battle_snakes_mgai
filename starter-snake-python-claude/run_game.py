import json
import os
import subprocess
import time
from pathlib import Path

MAX_TURNS = int(os.environ.get("MAX_TURNS", "300"))
LOG_PATH = Path(os.environ.get("GAME_LOG", "game.json"))
SEED = os.environ.get("SEED", "123")
OPEN_BROWSER = os.environ.get("BROWSER", "1") == "1"
MATCHUP = os.environ.get("MATCHUP", "mcts_vs_heuristic").strip().lower()

MATCHUPS = {
    "mcts_vs_heuristic": [
        ("MCTS", "http://127.0.0.1:8000"),
        ("Heuristic", "http://127.0.0.1:8001"),
    ],
    "mcts_vs_random": [
        ("MCTS", "http://127.0.0.1:8000"),
        ("Random", "http://127.0.0.1:8001"),
    ],
    "heuristic_vs_random": [
        ("Heuristic", "http://127.0.0.1:8000"),
        ("Random", "http://127.0.0.1:8001"),
    ],
}


def build_cmd():
    players = MATCHUPS.get(MATCHUP)
    if players is None:
        valid = ", ".join(sorted(MATCHUPS))
        raise ValueError(f"Unknown MATCHUP '{MATCHUP}'. Valid options: {valid}")

    cmd = [
        "battlesnake",
        "play",
        "-W",
        "11",
        "-H",
        "11",
        "-g",
        "standard",
        "-m",
        "hz_hazard_pits",
    ]

    for name, url in players:
        cmd.extend(["--name", name, "--url", url])

    cmd.extend(
        [
            "--foodSpawnChance",
            "25",
            "--minimumFood",
            "2",
            "--seed",
            SEED,
            "--timeout",
            "1000",
            "--output",
            str(LOG_PATH),
        ]
    )

    if OPEN_BROWSER:
        cmd.append("--browser")

    return cmd


def load_last_state(path: Path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file_handle:
        lines = file_handle.read().splitlines()

    if not lines:
        return None

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "turn" in payload:
            return payload

    return None


def print_final_summary(last_state):
    snakes = sorted(
        last_state["board"]["snakes"],
        key=lambda snake: (-snake["length"], snake["name"]),
    )
    print("\nFinal survivors")
    for snake in snakes:
        print(
            f"{snake['name']}: length={snake['length']} health={snake['health']}"
        )


def main():
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    cmd = build_cmd()
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd)

    last_turn = -1
    last_state = None

    try:
        while proc.poll() is None:
            state = load_last_state(LOG_PATH)
            if state is not None:
                last_state = state
                turn = int(state.get("turn", -1))
                if turn != -1 and turn != last_turn:
                    last_turn = turn
                    print(f"turn={turn}")

                if turn >= MAX_TURNS:
                    print(f"Reached cap at turn {turn}. Stopping game.")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break

            time.sleep(0.1)
    finally:
        if proc.poll() is None:
            proc.kill()

    if last_state is None:
        print("Game ended before a final state was captured.")
        return

    print_final_summary(last_state)


if __name__ == "__main__":
    main()
