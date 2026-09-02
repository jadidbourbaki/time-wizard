# time-wizard

A 450M parameter vision-language model fine-tuned to read analog clocks
in photographs, measured against frontier models on the same 200 held-out
photos.

## The task

The model receives a cropped photograph of a clock and one question,
"What time does this analog clock show?", and answers with hours and
minutes as JSON. A prediction counts as correct when it is
within one minute of the label on the 12 hour dial, wraparound
included. That is what the paper's own `eval.py` computes for its top-1
metric, so numbers published on this data sit on the same scale as
ours. The score also reports exact matches, hour accuracy, mean
circular error in minutes, and how many replies had no parseable time.

Published numbers on this data, for context:

| System | Top-1 | Source |
|---|---|---|
| Specialized ResNet-50 with a spatial transformer | 80.4 COCO, 77.3 OpenImages | It's About Time, Table 3, end to end |
| The same model on detector crops with IoU above 50 | 82.9 COCO, 79.6 OpenImages | It's About Time, Table 5 |
| Molmo-7B-D | 68.2 | Molmo, Table 10 |
| MolmoE-1B | 65.8 | Molmo, Table 10 |
| Qwen2-VL-72B | 9.1 | Molmo, Table 10 |
| Claude 3.5 Sonnet | 6.6 | Molmo, Table 10 |
| GPT-4o | 2.7 | Molmo, Table 10 |

Read those as context, not as rows in the same table. Molmo's column
averages three test sets including Clock Movies, which we exclude, and
every row covers the full sets rather than our 200 photos. Molmo used
the prompt "What time is being shown?" and the same evaluation protocol
from the itsabouttime repo.

## Data

Labels come from It's About Time (Yang and Zisserman, 2022): 3228 clocks
in COCO and OpenImages photographs with human hour and minute labels
and detector boxes, published as CSVs in the itsabouttime repo.
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
[our fork](https://github.com/jadidbourbaki/itsabouttime), are available
as extra training data for the synthetic-only ablation.

## What the numbers mean

The claim is about specialization: a model fine-tuned on about 2800
labelled photos against frontier models given none. The frontier models
have plausibly seen these COCO and OpenImages photos in pretraining and
the label CSVs have been public since 2022, so contamination runs in
their favour if anywhere. Planned ablations: train on COCO and test on
OpenImages and the reverse, train on SynClock renders alone, and give
the frontier models a few labelled crops in context.

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

Training needs a CUDA GPU. `sky/train.yaml` provisions one on Nebius,
syncs the repo, rebuilds the crops from their seed, fine-tunes, and scores
the result. See `sky/README.md` for credentials, then:

```
just sky-train
just sky-fetch
just sky-down
```

The first line fine-tunes on the 2796 photo crops with eval loss on the
dev split each epoch. The second trains on rendered clocks only. Both
write `config.json`, checkpoints, and the LoRA adapter under the output
directory. `just train --help` lists every option. Needs a CUDA GPU.
Colab Pro is enough for the 450M model.

## Evaluation

```
just bench --adapter runs/tw-photos/adapter --split dev
just bench --model anthropic:claude-fable-5-1 --split dev --limit 20
AWS_REGION=us-east-1 just bench --model bedrock-mantle:openai.gpt-5.6-sol --split dev
```

`--adapter` scores the local model, `--model` any pydantic-ai model id
at `--effort max` by default, and `--limit` scores a prefix of the split
for a cheap check. Claude reads `ANTHROPIC_API_KEY`, Bedrock reads
`AWS_BEARER_TOKEN_BEDROCK` or the AWS credential chain.

Each reply is appended to `runs/bench/<split>_<model>.jsonl` the moment
it arrives, and the score lands beside it as `.score.json`. Rerunning
the same command answers only the clocks missing from that file, so an
interrupted run resumes and no API spend is repeated. Delete the file to
force a fresh run. A clock the model cannot answer records a bracketed
reason, counts as unparsed, and never aborts the run. The test split is
graded once per final model.
