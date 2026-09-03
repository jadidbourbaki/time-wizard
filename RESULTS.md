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
dev score guides tuning. The test split is scored once, after tuning
stops.

| Run | Split | Best dev loss | Within 1 min | Exact | Hour correct | Mean error | Unreadable |
|---|---|---|---|---|---|---|---|
| run 1, lr 2e-5, 3 epochs, best epoch 2 | dev | 0.346 | 25.0% | 9.0% | 44.0% | 76 min | 0 |
| lr 5e-5, 3 epochs | dev | 0.545 | 1.5% | | 12.5% | 151 min | |
| lr 1e-5, 3 epochs | dev | 0.489 | 5.5% | | 17.5% | 137 min | |
| lr 1e-5, 5 epochs | dev | 0.415 | not scored | | | | |
| lr 5e-5, 5 epochs | dev | 0.290 at epoch 2, then cancelled | | | | | |
| final | test | | | | | |

Every run trains in 90 to 160 seconds on one H100. Run 1 produced valid
JSON on every photograph. Its errors are hours off rather than minutes
off, which points at confusing the two hands.

The learning rate sweep on 2026-09-03 was cut short to stop paying for
the machine. Two findings survive it. Dev loss tracks the score: 0.55
gave 1.5 percent, 0.49 gave 5.5 percent, 0.35 gave 25 percent. And the
three epoch runs are unstable in the rate, since both neighbours of 2e-5
did far worse, while 5e-5 over five epochs reached the lowest dev loss
seen before it was cancelled. The next session should finish the five
epoch runs at 5e-5 and 2e-5 and try a second seed at 2e-5 to measure
run to run variance.

Four in five COCO clocks are under 224 pixels across in the source
photograph. At that size a minute of arc is under a pixel, which caps
every model on this benchmark. The OpenImages clocks are larger and
would gain from a higher resolution rebuild of the crops.
