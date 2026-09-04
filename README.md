# Time Wizard

Time Wizard is a 450 million parameter vision language model that reads
analog clocks in photographs. This repository fine-tunes the model,
scores frontier models on the same 200 held out photographs, and
publishes the benchmark, the model, and every reply. RESULTS.md holds the
full tables.

## Task and metric

The model sees one photograph cropped to a clock. The model answers the
question "What time does this analog clock show?" with hours and minutes
as JSON.

A prediction counts as correct when the prediction lands within one
minute of the label. Distance wraps around the 12 hour dial. A prediction
of 12:59 against a label of 1:00 is therefore one minute of error.

The score carries four more numbers: exact matches, hour accuracy, mean
error in minutes, and the count of replies that held no readable time.

## Data

Yang, Xie, and Zisserman labelled 3228 clocks for their 2022 paper "It's
About Time". The clocks appear in photographs from COCO and OpenImages.
The authors read each clock to the minute by eye. An object detector
marked each clock with a box. The paper's repository publishes the labels
and the boxes as CSV files.

`timewizard.photos build` turns those labels into splits:

1. Download each photograph from COCO or OpenImages.
2. Crop around the detector box with a 20 percent margin on each side.
   The paper uses the same margin.
3. Pad the crop to a square and resize it to 448 pixels.
4. Drop near duplicates. A perceptual hash gives every crop a short
   fingerprint. Two crops with close fingerprints show the same clock.
5. Split by a fixed seed. Every split holds the same share of COCO
   photographs as the whole pool.

| Split | Clocks | Use |
|---|---|---|
| test | 200 | graded once per final model |
| dev | 200 | hyperparameter choices and checkpoint selection |
| train | 2796 | fine-tuning |

`benchmark/` holds the three splits as image ids and labels. The split
files are the record of which photograph belongs where. The pixels live
in `data/` under the licences of COCO and OpenImages.

The five steps above take about twenty minutes. Every machine skips them
by pulling the finished crops instead. `just photos push` uploads the
crops to the Hugging Face dataset `jadidbourbaki/time-wizard-bench`.
`just photos pull` fetches them in about two minutes. `just photos build`
reruns the five steps from the source images. Use `build` to regenerate
the dataset or to check the uploaded copy against the originals.
`benchmark/README.md` is the dataset card shown on the Hub. `just photos
card` uploads the card on its own after an edit. `push` uploads the card
after the crops.

The dataset is public. OpenImages photographs carry CC BY 2.0. COCO
photographs come from Flickr under the licence each photographer chose.
The dataset card states both and names the source image of every crop.

The base model is
[LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M).

## Scope of the comparison

The fine-tuned model studies 2796 labelled photographs. The frontier
models study none. The comparison measures what fine-tuning buys on one
narrow task. The comparison says nothing about general capability.

COCO and OpenImages have been public for years. The label files have been
public since 2022. Frontier models may have read all of that during
pretraining. Any such contamination helps them.

Two ablations remain to run. Train on COCO and test on OpenImages, then
reverse the two. Give the frontier models a few labelled crops before the
question.

## Setup

```
just setup
just photos pull
```

`setup` runs `uv sync`. `uv sync` installs the pinned runtime, training,
and baseline dependencies. `photos pull` downloads the 3196 finished crops
from the Hugging Face dataset in about two minutes. Uploading the crops,
the cards, or the model needs a Hugging Face login:

```
uv run hf auth login
```

```
just check
```

`check` runs the formatter, the linter, the type checker, and the tests.
The tests need no GPU. The tests reach no network.

## Reading a clock

```
just read clock.jpg
```

`read` loads the fine-tuned model from the Hub and prints the time the
model reads as JSON, such as `{"hours":3,"minutes":28}`. Crop the
photograph to the clock first. The reader pads the crop to a square and
resizes the square to 448 pixels, the shape the model trained on.
`--checkpoint` points the reader at a local directory or another Hub
repo. In Python, `timewizard.reader.Reader` loads the model once. The `read`
method of `Reader` returns a `Time` or None for each image.

## Training

```
just train --out runs/tw-photos
```

`train` fine-tunes every weight of the model on the 2796 photograph crops
for five epochs at a learning rate of 5e-5, the setting that won the
sweep in RESULTS.md. The trainer measures loss on the dev split after
every epoch and keeps the epoch with the lowest loss. The trainer writes
`config.json`, the checkpoints, and the final model under the output
directory. The trainer then uploads the final model to the Hugging Face
repo `jadidbourbaki/time-wizard`. A scoring run anywhere can therefore
load the model by name. `just train --help` lists every option.

Every weight trains rather than a low rank adapter. The model has 450
million parameters and fits one GPU with room to spare. Reading the angle
of a minute hand asks the vision encoder to change. An adapter on the
attention layers changes the encoder less.

`model/README.md` is the model card shown on the Hub. `just figures`
draws the card's figures from the score files in `runs/bench/`. `just
model-card` uploads the card and the figures.

### Nebius

Training needs a CUDA GPU. `sky/train.yaml` describes the whole run.
[SkyPilot](https://docs.skypilot.ai) reads that file and provisions one
H100. SkyPilot installs uv on the machine, copies the working directory
across, pulls the crops from the Hub, fine-tunes the model, and scores the
result on the dev split. The upload from this machine stays the size of
this repository.

Install SkyPilot once as a tool. A throwaway environment will not do,
because SkyPilot runs a local API server that must outlive the command
that started it.

```
uv tool install --with pip "skypilot[nebius]"
sky api start
```

SkyPilot then reads Nebius credentials from two files. The Nebius CLI
writes both:

```
mkdir -p ~/.nebius
nebius iam get-access-token > ~/.nebius/NEBIUS_IAM_TOKEN.txt
nebius --format json iam whoami | jq -r '.user_profile.tenants[0].tenant_id' > ~/.nebius/NEBIUS_TENANT_ID.txt
sky check nebius
```

The last command prints `Nebius: enabled` on success. The access token
expires after a few hours. `just sky-train` therefore rewrites that file
before every launch.

```
just sky-train
just sky-fetch
just sky-down
```

The task names one instance type, `gpu-h100-sxm_1gpu-16vcpu-200gb`, at
$3.85 an hour. Nebius created L40S machines for us on both host platforms
and started none of them. The H100 started first time. A full training
run takes under three minutes. The dearer GPU therefore costs cents more
per run. Run `sky gpus list --infra nebius` to see the options before
changing the instance type.

The task sets `TORCH_DISABLE_NATIVE_JIT=1`. torch 2.14 otherwise routes
some LFM2 operations through Triton kernels. Triton compiles a stub
against Python's C headers. The machine's system Python lacks those
headers. The flag keeps torch on its ordinary kernels.

`sky-train` provisions the GPU and runs the job. `sky-train` reads your
Hugging Face token from `~/.cache/huggingface/token`, where `hf auth
login` put it, and passes the token to the machine as a secret. The
machine needs the token to upload the finished model. `sky-fetch` copies
`runs/` back to this machine. `sky-down` releases the GPU. The task also
shuts the machine down on its own after fifteen idle minutes. A forgotten
cluster therefore costs at most a quarter of an hour. The model is
already on the Hub by then.

One command changes the training flags without touching the committed
file:

```
sky launch -c time-wizard sky/train.yaml --env TRAIN_ARGS="--out runs/five-epochs --epochs 5"
```

## Evaluation

```
just bench --checkpoint jadidbourbaki/time-wizard --split dev
just bench --model anthropic:claude-fable-5-1 --split dev --limit 20
AWS_REGION=us-east-1 just bench --model bedrock-mantle:openai.gpt-5.6-sol --split dev
```

`--checkpoint` scores our fine-tuned model from a local directory or a
Hugging Face repo. `--model` scores any model that pydantic-ai can reach,
at maximum reasoning effort by default. `--limit` scores the first few
photographs of a split. Use `--limit` as a cheap check before a full run.
Claude reads `ANTHROPIC_API_KEY`. Bedrock reads `AWS_BEARER_TOKEN_BEDROCK`
and falls back to the AWS credential chain.

The runner writes each reply to `runs/bench/<split>_<model>.jsonl` as the
reply arrives, with the tokens the reply cost. The score lands beside the
replies with a `.score.json` suffix. A second run of the same command
asks only about the photographs missing from that file. An interrupted
run therefore resumes and repeats no API spend. Delete the file to force
a fresh run.

A photograph the model cannot answer records a bracketed reason. The
score counts that photograph as unreadable. One such photograph never
stops the run.

Grade the test split once per final model. `RESULTS.md` records every
score. `runs/bench/` keeps every test reply under version control.
