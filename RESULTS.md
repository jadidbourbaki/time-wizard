# Results

All numbers come from the 200 photograph test split in `benchmark/`. Every
model sees the same 448 pixel crops and the same prompt from
`timewizard.reading`. A prediction is correct when it lands within one
minute of the label on the 12 hour dial.

## Frontier models

Scored through `timewizard.bench` at maximum reasoning effort with a
budget of 32000 tokens per clock. Opus 5 ran on 2026-09-03. GPT-5.6 Sol
and Fable 5.1 ran on 2026-09-04. Replies and scores sit in `runs/bench/`.

| Model | Within 1 min | Exact | Hour | Mean error | Unreadable | Input tokens | Output tokens | Cost |
|---|---|---|---|---|---|---|---|---|
| GPT-5.6 Sol, Bedrock | 61.5% | 31.0% | 68.5% | 65 min | 0 | 79k | 686k | $14.04 |
| Claude Fable 5.1 | 21.5% | 11.5% | 42.5% | 107 min | 2 | 66k | 972k | $49.25 |
| Claude Opus 5 | 15.5% | 6.0% | 31.0% | 141 min | 0 | 66k | 366k | $9.47 |

Mean error is the circular distance in minutes averaged over the readable
replies. Unreadable counts replies with no valid time and scores them as
wrong. Output tokens include reasoning. Cost multiplies the token counts
by the list prices on the day of the run: $4 and $20 per million input
and output tokens for GPT-5.6 Sol on Bedrock, $10 and $50 for Fable 5.1,
and $5 and $25 for Opus 5. Fable 5.1 spent its whole 32000 token budget
on two clocks without answering. The runner records no tokens for a
failed request. The two failed requests therefore add about 64k output tokens
and $3.20 that the table leaves out.

An earlier pass over the same split on 2026-09-03 scored GPT-5.6 Sol at
60.5 percent and Fable 5.1 at 24.0 percent. The earlier pass lost its token
counts to an interruption. We reran both models and replaced the earlier
replies.

## Time Wizard

Every run fine-tunes every weight of LFM2.5-VL-450M on the 2796
photograph train split and keeps the epoch with the lowest dev loss. The
dev score guided tuning. We scored the test split once, on the run with
the best dev score.

### Test split, 200 photographs

| Model | Effort | Within 1 min | Exact | Hour | Mean error | Unreadable |
|---|---|---|---|---|---|---|
| Time Wizard, 450M | none | 59.0% | 31.0% | 74.5% | 44 min | 0 |
| GPT-5.6 Sol, Bedrock | max | 61.5% | 31.0% | 68.5% | 65 min | 0 |
| Claude Fable 5.1 | max | 21.5% | 11.5% | 42.5% | 107 min | 2 |
| Claude Opus 5 | max | 15.5% | 6.0% | 31.0% | 141 min | 0 |

On the headline metric the 450M model and GPT-5.6 Sol are within
sampling error of each other. With 200 photographs the standard error
near 60 percent is about 3.5 points. The two differ by 2.5. The small
model reads the hour correctly more often and misses by less on average.
The small model is 38 points above Claude Fable 5.1 and 44 above Opus 5.

The final model is `jadidbourbaki/time-wizard` on the Hub. Training
took 142 seconds on one H100 with learning rate 5e-5 over five epochs.
Dev loss reached a minimum of 0.249 after epoch three. The epoch three
checkpoint is the one scored above. The row above comes from a rerun of
that checkpoint on a second H100 on 2026-09-04. The rerun read the 200
clocks in 26 seconds. The rerun matched the original run on every number
except exact matches, where 31.0 percent replaced 31.5. One clock's
minute shifted under bfloat16 kernel noise. The rerun's replies are the
ones in `runs/bench/`.

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
across every run. Dev loss is therefore a sound selection criterion. Two
seeds at the same setting landed half a point apart. Run to run variance
is therefore small and the gains below are real. The schedule and the
rate both mattered. Five epochs lifted 2e-5 from 28 to 38 percent. The
rate 5e-5 on that longer schedule reached 60.5. The same rate over three
epochs collapsed to 1.5 percent. The longer warmup and slower decay are
what make the higher rate usable.

Four in five COCO clocks are under 224 pixels across in the source
photograph. At that size a minute of arc is under a pixel. The resolution
caps every model on this benchmark. The OpenImages clocks are larger and
would gain from a higher resolution rebuild of the crops.
