# Results

All numbers come from the 200 photograph test split in `benchmark/`. Every
model sees the same 448 pixel crops and the same prompt from
`timewizard.reading`. A prediction is correct when it lands within one
minute of the label on the 12 hour dial.

## Frontier models

Scored on 2026-09-03 through `timewizard.bench` at maximum reasoning
effort. Replies and scores sit in `runs/bench/`.

| Model | Within 1 min | Exact | Hour correct | Mean error | Unreadable | Output tokens |
|---|---|---|---|---|---|---|
| GPT-5.6 Sol, Bedrock | 60.5% | 32.0% | 68.5% | 59 min | 1 | 88k |
| Claude Fable 5.1 | 24.0% | 9.5% | 40.0% | 119 min | 0 | 124k |
| Claude Opus 5 | 15.5% | 6.0% | 31.0% | 141 min | 0 | 366k |

Mean error is the circular distance in minutes averaged over the readable
replies. Unreadable counts replies with no valid time and scores them as
wrong. Output tokens include reasoning.

## time-wizard

Pending. The first run fine-tunes every weight of LFM2.5-VL-450M on the
2796 photograph train split for three epochs and keeps the epoch with the
lowest dev loss. The dev score guides any tuning. The test split is scored
once, after tuning stops.

| Model | Split | Within 1 min | Exact | Hour correct | Mean error | Unreadable |
|---|---|---|---|---|---|---|
| time-wizard, run 1 | dev | | | | | |
| time-wizard, final | test | | | | | |
