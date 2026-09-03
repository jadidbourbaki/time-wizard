# time-wizard

A 450M parameter vision-language model reads analog clocks in
photographs. This repository fine-tunes that model. It then scores the
model against frontier models on the same 200 held-out photographs.

## Task and metric

The model sees one cropped photograph of a clock. It answers a single
question: "What time does this analog clock show?" It replies with hours
and minutes as JSON.

A prediction counts as correct when it lands within one minute of the
label. Distance wraps around the 12 hour dial. A prediction of 12:59
against a label of 1:00 is therefore one minute of error.

The score carries four more numbers: exact matches, hour accuracy, mean
error in minutes, and the count of replies that held no readable time.

## Data

Yang and Zisserman labelled 3228 clocks for their 2022 paper "It's About
Time". The clocks appear in photographs from COCO and OpenImages. Human
annotators read each clock to the minute. An object detector marked each
clock with a box. Their repository publishes the labels and the boxes as
CSV files.

`timewizard.photos` turns those labels into splits:

1. Download each photograph from COCO or OpenImages.
2. Crop around the detector box, keeping a 20 percent margin on each
   side. The paper uses the same margin.
3. Pad the crop to a square.
4. Drop near duplicates. A perceptual hash gives every crop a short
   fingerprint. Two crops with close fingerprints show the same clock.
5. Split by a fixed seed. Every split holds the same share of COCO
   photographs as the whole pool.

| Split | Clocks | Use |
|---|---|---|
| test | 200 | graded once per final model |
| dev | 200 | hyperparameter choices and training loss |
| train | 2796 | fine-tuning |

`benchmark/` holds the three splits as image ids and labels. The pixels
live in `data/` under the licences of COCO and OpenImages. `just photos`
rebuilds them from the seed.

The base model is
[LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M).

## Scope of the comparison

The fine-tuned model studies 2796 labelled photographs. The frontier
models study none. This measures what fine-tuning buys on one narrow
task. It says nothing about general capability.

COCO and OpenImages have been public for years. The label files have been
public since 2022. Frontier models may have read all of it during
pretraining. Any such contamination helps them.

Two ablations remain to run. Train on COCO and test on OpenImages, then
reverse the two. Give the frontier models a few labelled crops before the
question.

## Setup

```
just setup
just photos
```

`setup` runs `uv sync`. That installs the pinned runtime, training, and
baseline dependencies. `photos` downloads the label files and about 3200
images. It writes the crops and the splits. Running it twice changes
nothing, because the seed fixes the result.

```
just check
```

This runs the formatter, the linter, the type checker, and the tests. The
tests need no GPU. They reach no network.

## Training

```
just train --out runs/tw-photos --epochs 3
```

This fine-tunes on the 2796 photograph crops. It measures loss on the dev
split after every epoch. It writes `config.json`, the checkpoints, and the
trained weights under the output directory. `just train --help` lists
every option.

### Nebius

Training needs a CUDA GPU. `sky/train.yaml` describes the whole run.
[SkyPilot](https://docs.skypilot.ai) reads that file and provisions one
L40S. It installs uv on the machine. It copies the working directory
across. It rebuilds the crops from the seed. It fine-tunes the model. It
scores the result on the dev split. The crops never travel over the
network. The upload stays the size of this repository.

SkyPilot reads Nebius credentials from two files. The Nebius CLI writes
both:

```
uv tool install --with pip "skypilot[nebius]"
mkdir -p ~/.nebius
nebius iam get-access-token > ~/.nebius/NEBIUS_IAM_TOKEN.txt
nebius --format json iam whoami | jq -r '.user_profile.tenants[0].tenant_id' > ~/.nebius/NEBIUS_TENANT_ID.txt
sky check nebius
```

The last command prints `Nebius: enabled` on success. The access token
expires after a few hours. Write the first file again when a launch fails
to authenticate.

```
just sky-train
just sky-fetch
just sky-down
```

`sky-train` provisions the GPU and runs the job. `sky-fetch` copies
`runs/` back to this machine. `sky-down` releases the GPU. Nebius bills
for every hour the machine stays up. Fetch the trained weights, then shut
it down.

One command changes the training flags without touching the committed
file:

```
sky launch -c time-wizard sky/train.yaml --env TRAIN_ARGS="--out runs/five-epochs --epochs 5"
```

## Evaluation

```
just bench --adapter runs/tw-photos/adapter --split dev
just bench --model anthropic:claude-fable-5-1 --split dev --limit 20
AWS_REGION=us-east-1 just bench --model bedrock-mantle:openai.gpt-5.6-sol --split dev
```

`--adapter` scores our fine-tuned model. `--model` scores any model that
pydantic-ai can reach, at maximum reasoning effort by default. `--limit`
scores the first few photographs of a split. Use it as a cheap check
before a full run. Claude reads `ANTHROPIC_API_KEY`. Bedrock reads
`AWS_BEARER_TOKEN_BEDROCK`. It falls back to the AWS credential chain.

The runner writes each reply to `runs/bench/<split>_<model>.jsonl` as it
arrives. The score lands beside it with a `.score.json` suffix. A second
run of the same command asks only about the photographs missing from that
file. An interrupted run therefore resumes. It repeats no API spend.
Delete the file to force a fresh run.

A photograph the model cannot answer records a bracketed reason. The
score counts that photograph as unreadable. One such photograph never
stops the run.

Grade the test split once per final model.
