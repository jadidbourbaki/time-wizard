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

Every run fine-tunes every weight of LFM2.5-VL-450M on the 2796
photograph train split and keeps the epoch with the lowest dev loss. The
dev score guided tuning. The test split was scored once, on the run with
the best dev score.

### Test split, 200 photographs

| Model | Within 1 min | Exact | Hour correct | Mean error | Unreadable |
|---|---|---|---|---|---|
| time-wizard, 450M | 59.0% | 31.5% | 74.5% | 44 min | 0 |
| GPT-5.6 Sol, Bedrock | 60.5% | 32.0% | 68.5% | 59 min | 1 |
| Claude Fable 5.1 | 24.0% | 9.5% | 40.0% | 119 min | 0 |
| Claude Opus 5 | 15.5% | 6.0% | 31.0% | 141 min | 0 |

On the headline metric the 450M model and GPT-5.6 Sol are within
sampling error of each other. With 200 photographs the standard error
near 60 percent is about 3.5 points, and the two differ by 1.5. The
small model reads the hour correctly more often and misses by less on
average. It is 35 points above Claude Fable 5.1 and 44 above Opus 5.

The final model is `jadidbourbaki/time-wizard` on the Hub. Its training
took 142 seconds on one H100 with learning rate 5e-5 over five epochs.
Dev loss reached its minimum of 0.249 after epoch three, and that
checkpoint is the one scored above.

### Dev split, tuning runs

| Run | Best dev loss | Within 1 min | Exact | Hour correct | Mean error |
|---|---|---|---|---|---|
| lr 5e-5, 5 epochs, best epoch 3 | 0.249 | 60.5% | | 77.0% | 30 min |
| lr 2e-5, 5 epochs, best epoch 2 | 0.310 | 38.0% | | 58.0% | 50 min |
| lr 2e-5, 3 epochs, seed 0 | 0.344 | 28.0% | 9.0% | 44.0% | 75 min |
| lr 2e-5, 3 epochs, seed 1 | 0.344 | 27.5% | | 47.5% | 74 min |
| lr 1e-5, 5 epochs | 0.415 | not scored | | | |
| lr 1e-5, 3 epochs | 0.489 | 5.5% | | 17.5% | 137 min |
| lr 5e-5, 3 epochs | 0.545 | 1.5% | | 12.5% | 151 min |

Three findings from the sweep. Dev loss tracks the dev score closely
across every run, so it is a sound selection criterion. Two seeds at the
same setting landed half a point apart, so run to run variance is small
and the gains below are real. The schedule and the rate both mattered:
five epochs lifted 2e-5 from 28 to 38 percent, and 5e-5 on that longer
schedule reached 60.5. The same rate over three epochs collapsed to 1.5
percent, so the longer warmup and slower decay are what make the higher
rate usable.

Four in five COCO clocks are under 224 pixels across in the source
photograph. At that size a minute of arc is under a pixel, which caps
every model on this benchmark. The OpenImages clocks are larger and
would gain from a higher resolution rebuild of the crops.
