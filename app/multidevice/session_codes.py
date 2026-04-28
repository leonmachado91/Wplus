"""Generates short memorable session codes like 'mesa-4221'."""
import random
import secrets

_WORDS = [
    "mesa", "sala", "team", "live", "talk", "crew", "meet", "sync",
    "alfa", "beta", "echo", "golf", "delta", "bravo", "novo", "link",
]


def generate_code() -> str:
    word = random.choice(_WORDS)
    number = secrets.randbelow(9000) + 1000  # 1000–9999
    return f"{word}-{number}"
