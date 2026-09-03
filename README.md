# time-wizard

A 450M parameter vision-language model fine-tuned to read analog clocks
in photographs, measured against frontier models on the same 200 held-out
photos.

## Task and metric

The model receives a cropped photograph of a clock and one question,
"What time does this analog clock show?", and answers with hours and
minutes as JSON. A prediction counts as correct when it is within one
minute of the label on the 12 hour dial, wraparound included. The score
also reports exact matches, hour accuracy, mean circular error in
minutes, and how many replies had no parseable time.

## Data

Labels come from It's About Time (Yang and Zisserman, 2022): 3228 clocks
in COCO and OpenImages photographs with human hour and minute labels and
detector boxes, published as CSVs in the itsabouttime repo.
`timewizard.photos` fetches the images from the source datasets, crops
around the detector box with the paper's 20 percent margin, pads to a
square, drops near duplicates by perceptual hash, and splits by a fixed
seed, stratified by source.

| Split | Clocks | Use |
|---|---|---|
| test | 200 | graded once per final model |
| dev | 200 | hyperparameter selection and eval loss |
| train | 2796 | fine-tuning |

`benchmark/` holds the three splits as image ids and labels. Pixels stay
in `data/` under the licences of COCO and OpenImages and are rebuilt by
`just photos`. The base model is
[LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M).
Rendered clocks from SynClock, installed from
[our fork](https://github.com/jadidbourbaki/itsabouttime), are extra
training data for the synthetic-only ablation.

## Scope of the comparison

The fine-tuned model sees 2796 labelled photos and the frontier models
see none, so the result measures specialization rather than general
capability. COCO and OpenImages photographs and the label CSVs have been
public since 2022, so any contamination favours the frontier models.

Ablations: train on COCO and test on OpenImages and the reverse, train on
SynClock renders alone, and give the frontier models labelled crops in
context.

## Setup

```
just setup
just photos
```

`setup` runs `uv sync`, which installs the pinned runtime, training, and
baseline dependencies. `photos` downloads the label CSVs and about 3200
images, crops, deduplicates, and writes the splits. Rerunning it is
idempotent and reproduces the same splits from the seed.

```
just check
```

Runs the formatter in check mode, the linter, the type checker, and the
tests. The tests need no GPU and no network.

## Training

```
just train --out runs/tw-photos --epochs 3
just train --out runs/tw-rendered --photos false --rendered 100000
```

The first line fine-tunes on the 2796 photo crops with eval loss on the
dev split each epoch. The second trains on rendered clocks only. Both
write `config.json`, checkpoints, and the LoRA adapter under the output
directory. `just train --help` lists every option.

### Nebius

Training needs a CUDA GPU. `sky/train.yaml` describes the whole run, so
[SkyPilot](https://docs.skypilot.ai) provisions one L40S, installs uv,
syncs the working directory, rebuilds the crops from their seed,
fine-tunes, and scores the adapter on the dev split. Training data is
never uploaded, so the transfer is the repo alone.

SkyPilot reads Nebius credentials from two files written by the Nebius
CLI:

```
uv tool install --with pip "skypilot[nebius]"
mkdir -p ~/.nebius
nebius iam get-access-token > ~/.nebius/NEBIUS_IAM_TOKEN.txt
nebius --format json iam whoami | jq -r '.user_profile.tenants[0].tenant_id' > ~/.nebius/NEBIUS_TENANT_ID.txt
sky check nebius
```

The last command prints `Nebius: enabled` when it works. The IAM token
expires, so rewrite the first file before a launch that fails to
authenticate.

```
just sky-train
just sky-fetch
just sky-down
```

`sky-train` provisions the GPU and runs the job, `sky-fetch` copies
`runs/` back, and `sky-down` releases the GPU. The cluster bills while it
is up, so take the adapter and stop it. To change training flags without
editing the committed task:

```
sky launch -c time-wizard sky/train.yaml --env TRAIN_ARGS="--out runs/rendered --photos false --rendered 100000"
```

## Evaluation

```
just bench --adapter runs/tw-photos/adapter --split dev
just bench --model anthropic:claude-fable-5-1 --split dev --limit 20
AWS_REGION=us-east-1 just bench --model bedrock-mantle:openai.gpt-5.6-sol --split dev
```

`--adapter` scores the local model, `--model` any pydantic-ai model id at
`--effort max` by default, and `--limit` scores a prefix of the split for
a cheap check. Claude reads `ANTHROPIC_API_KEY`, Bedrock reads
`AWS_BEARER_TOKEN_BEDROCK` or the AWS credential chain.

Each reply is appended to `runs/bench/<split>_<model>.jsonl` the moment
it arrives, and the score lands beside it as `.score.json`. Rerunning the
same command answers only the clocks missing from that file, so an
interrupted run resumes and no API spend is repeated. Delete the file to
force a fresh run. A clock the model cannot answer records a bracketed
reason, counts as unparsed, and never aborts the run. The test split is
graded once per final model.
