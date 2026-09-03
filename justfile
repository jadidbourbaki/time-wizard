set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# Create the environment. Run once after cloning.
setup:
    uv sync

# Build the photo benchmark: download, crop, deduplicate, split.
photos *args:
    uv run python -m timewizard.photos {{args}}

# Fine-tune. Example: just train --out runs/tw-photos
train *args:
    uv run python -m timewizard.train {{args}}

# Score a model on a split. Example: just bench --adapter runs/tw-photos/adapter --split dev
bench *args:
    uv run python -m timewizard.bench {{args}}

fmt:
    uv run ruff format .

check:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check timewizard tests
    uv run pytest -q

# Provision a Nebius GPU and run the fine-tune. See README.md.
sky-train *args:
    uv run --with "skypilot[nebius]" sky launch -c time-wizard sky/train.yaml {{args}}

# Copy runs/ back from the Nebius box.
sky-fetch:
    uv run --with "skypilot[nebius]" sky rsync down time-wizard ~/sky_workdir/runs runs

# Release the Nebius GPU.
sky-down:
    uv run --with "skypilot[nebius]" sky down time-wizard
