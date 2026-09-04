set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# Create the environment. Run once after cloning.
setup:
    uv sync

# Fetch, rebuild, or upload the photo crops and card: pull, build, push, card.
photos *args:
    uv run python -m timewizard.photos {{args}}

# Read one clock photograph with the fine-tuned model. Example: just read clock.jpg
read *args:
    uv run python -m timewizard.reader {{args}}

# Fine-tune. Example: just train --out runs/tw-photos
train *args:
    uv run python -m timewizard.train {{args}}

# Draw the model card figures from the score files in runs/bench.
figures *args:
    uv run python -m timewizard.figures {{args}}

# Upload model/ as the model card on the Hub.
model-card:
    uv run hf upload jadidbourbaki/time-wizard model . --include "README.md" --include "*.png"

# Score a model on a split. Example: just bench --checkpoint jadidbourbaki/time-wizard --split dev
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
    nebius iam get-access-token > ~/.nebius/NEBIUS_IAM_TOKEN.txt
    sky launch -c time-wizard sky/train.yaml --secret HF_TOKEN="$(cat ~/.cache/huggingface/token)" {{args}}

# Copy runs/ back from the Nebius box.
sky-fetch:
    rsync -az time-wizard:sky_workdir/runs/ runs/

# Release the Nebius GPU.
sky-down:
    sky down time-wizard
