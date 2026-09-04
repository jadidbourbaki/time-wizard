---
library_name: transformers
pipeline_tag: image-text-to-text
base_model: LiquidAI/LFM2.5-VL-450M
base_model_relation: finetune
datasets:
  - jadidbourbaki/time-wizard-bench
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/LICENSE
language:
  - en
tags:
  - analog-clock
  - clock-reading
  - lfm2-vl
  - sft
model-index:
  - name: time-wizard
    results:
      - task:
          type: image-text-to-text
          name: Analog clock reading
        dataset:
          type: jadidbourbaki/time-wizard-bench
          name: time-wizard-bench
          split: test
        metrics:
          - type: accuracy
            name: Within 1 minute
            value: 59.0
          - type: accuracy
            name: Exact minute
            value: 31.5
          - type: accuracy
            name: Hour correct
            value: 74.5
---

# time-wizard

time-wizard reads the time from a photograph of an analog clock. It is a
450 million parameter vision language model that answers with the hour
and the minute as JSON. On 200 held out photographs it reads 59.0 percent
of clocks to within one minute and gets the hour right on 74.5 percent.
GPT-5.6 Sol at its highest reasoning setting reads 60.5 percent to within
one minute and gets the hour right on 68.5 percent of the same
photographs. Claude Fable 5.1 and Claude Opus 5 read 24.0 and 15.5
percent to within one minute.

The model is Liquid AI's
[LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) after
supervised fine-tuning on 2796 labelled clock photographs. The training
run took 142 seconds on one H100. The photographs, labels, and splits are
the public dataset
[time-wizard-bench](https://huggingface.co/datasets/jadidbourbaki/time-wizard-bench).
The training and scoring code is at
[github.com/jadidbourbaki/time-wizard](https://github.com/jadidbourbaki/time-wizard).

## Results on the test split

The test split holds 200 clock crops. No model saw them before scoring.
Every model received the same 448 pixel crop and the same question.

| Model | Parameters | Within 1 min | Exact | Hour correct | Mean error | Unreadable |
|---|---|---|---|---|---|---|
| time-wizard | 450M | 59.0% | 31.5% | 74.5% | 44 min | 0 |
| GPT-5.6 Sol | undisclosed | 60.5% | 32.0% | 68.5% | 59 min | 1 |
| Claude Fable 5.1 | undisclosed | 24.0% | 9.5% | 40.0% | 119 min | 0 |
| Claude Opus 5 | undisclosed | 15.5% | 6.0% | 31.0% | 141 min | 0 |

Within 1 min is the share of clocks read to within one minute on the 12
hour dial, with the distance wrapping at twelve. Exact is the share read
to the minute. Hour correct is the share with the right hour. Mean error
averages the wrapped distance in minutes over readable replies.
Unreadable counts replies with no valid time in them. The dataset card
gives the prompt, the scoring code, and the exact configuration of each
frontier model.

With 200 photographs the standard error near 60 percent is about 3.5
points. time-wizard and GPT-5.6 Sol are within sampling error of each
other on the one minute metric. time-wizard leads on the hour and on mean
error. Its gaps to Fable 5.1 and Opus 5 are 35 and 44 points.

## Three test clocks

| ![Gilded tower clock](https://huggingface.co/datasets/jadidbourbaki/time-wizard-bench/resolve/main/examples/coco_000000058397.png) | ![Red rimmed station clock](https://huggingface.co/datasets/jadidbourbaki/time-wizard-bench/resolve/main/examples/openimages_db298211fd461ee6.png) | ![Blurred church tower clock](https://huggingface.co/datasets/jadidbourbaki/time-wizard-bench/resolve/main/examples/coco_000000566236.png) |
|---|---|---|
| Label 3:28 | Label 1:20 | Label 1:00 |
| time-wizard 3:28 | time-wizard 1:20 | time-wizard 8:20 |
| GPT-5.6 Sol 3:28 | GPT-5.6 Sol 10:07 | GPT-5.6 Sol 1:30 |
| Claude Fable 5.1 4:25 | Claude Fable 5.1 1:23 | Claude Fable 5.1 3:40 |
| Claude Opus 5 4:28 | Claude Opus 5 10:22 | Claude Opus 5 7:10 |

The first two clocks are sharp. time-wizard reads both to the minute.
Two frontier models swap the hour and minute hands on the station clock
and answer 10:07 and 10:22. The third clock is a blur of a few dozen
pixels in the source photograph. Every model guesses, and every guess is
wrong.

## Perception versus reasoning

Reading an analog clock means judging the angle of two hands. One minute
moves the minute hand six degrees, so reading to the minute means
resolving angles to six degrees. The frontier models approach the task by
reasoning. They describe the hands in words, estimate the angles, check
which hand is longer, and verify the arithmetic. Each clock costs them
thousands of tokens. Their answers still land far from the label. Opus 5
misses by 141 minutes on average, which is close to the 180 minutes a
random guess would score.

time-wizard reads the clock in one forward pass and answers in a handful
of tokens. Its vision encoder learned from 2796 labelled examples to carry
the angle of each hand through to the language model, which turns the
angles into a time. The whole model is 450 million parameters and cost
about fifteen cents of GPU time to train. On this task that is enough to
match the best frontier model.

## Base model

[LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) is a
vision language model from Liquid AI built to run on phones and other
edge devices. It pairs a SigLIP2 NaFlex vision encoder of 86 million
parameters with LFM2.5-350M, a language model whose backbone mixes gated
short convolution blocks with grouped query attention blocks. The encoder
accepts images up to 512 by 512 pixels at their native aspect ratio and
turns each image into between 32 and 256 tokens. Liquid AI publishes the
model under the LFM Open License v1.0.

The base model follows instructions and describes images. It reads
clocks poorly. Recognition training rewards an encoder for knowing that
an object is a clock and rarely for knowing where its hands point.
Fine-tuning every weight lets the encoder learn to keep that detail.

## Usage

The repository provides a reader. Clone it, install with `just setup`,
and run:

```
just read clock.jpg
```

The reader loads the model from the Hub and prints the time it reads,
such as `{"hours":3,"minutes":28}`. Crop the photograph to the clock
first. In Python, load the model once and read many photographs:

```python
from PIL import Image
from timewizard.reader import Reader

reader = Reader()
time = reader.read(Image.open("clock.jpg"))
print(time.hours, time.minutes)
```

`read` returns None when the reply holds no valid time. The reader pads
the image to a square and resizes it to 448 pixels, the shape the model
trained on.

To use the weights without the repository, send one image and the fixed
prompt through transformers:

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image, ImageOps

SYSTEM = "Reply with ONLY the requested JSON, no preface and no code block."
PROMPT = 'What time does this analog clock show? Reply as JSON: {"hours": H, "minutes": M} with H from 1 to 12.'

processor = AutoProcessor.from_pretrained("jadidbourbaki/time-wizard", max_image_tokens=256)
model = AutoModelForImageTextToText.from_pretrained("jadidbourbaki/time-wizard", dtype=torch.bfloat16)

image = ImageOps.pad(Image.open("clock.jpg").convert("RGB"), (448, 448), color=(0, 0, 0))
messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]},
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)
out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0])
```

The system prompt and the question must match the strings above, because
the model saw only those during training. `max_image_tokens=256` is the
image budget used in training. A 448 pixel square occupies 196 image
tokens under it. Greedy decoding with 32 new tokens is enough for the
JSON reply.

## Training

The training data is the 2796 photograph train split of
time-wizard-bench. Each photograph is a crop around one clock from a COCO
or OpenImages scene, padded to a square and resized to 448 pixels, with a
human label for the hour and the minute. Each example becomes one
conversation: the system prompt, the image with the question, and the
label as the JSON reply. Nothing else went into training. There were no
synthetic clocks, no augmentation, and no other data. The test split was
never read during training or model selection.

Supervised fine-tuning continues training the base model on those
conversations with the objective it was pretrained on. The model predicts
each next token and pays a cross entropy penalty when it is wrong. The
loss covers the assistant reply only. Tokens of the system prompt, the
image, and the question are masked out, so the model learns to produce
the time and spends no capacity on reproducing the question.

Every weight trained, the vision encoder included. A low rank adapter
would have frozen the encoder and taught the language model to describe
what the encoder already sees. The encoder is what has to change. A 450
million parameter model fits on one GPU with room to spare, so full
fine-tuning costs nothing extra.

The optimiser is AdamW at a peak learning rate of 5e-5 with a cosine
decay and a warmup over the first three percent of steps. Each step sees
32 photographs, as two accumulated batches of 16. Training runs in
bfloat16 for five epochs, about 440 steps, with seed 0. After every
epoch the trainer measures loss on the 200 photograph dev split and keeps
the epoch with the lowest loss. The minimum of 0.249 came after epoch
three, and that checkpoint is the published model. The trainer is TRL's
SFTTrainer on transformers.

The learning rate and the number of epochs came from a sweep scored on
the dev split. The rate mattered a great deal. At 2e-5 the model reached
28 percent within one minute after three epochs and 38 percent after
five. At 5e-5 over three epochs the model collapsed to 1.5 percent. The
same rate over five epochs, with the longer warmup and the slower decay
that come with it, reached 60.5 percent. Dev loss tracked the dev score
across every run, so the lowest loss picked the best model without
touching the test split. Two seeds at the same setting landed half a
point apart, so the differences above are real.

The final run took 142 seconds on one H100 on Nebius. At the hourly
price of that machine the run cost about fifteen cents. The whole sweep
of seven runs took under twenty minutes of GPU time.

## Limitations

The model reads a clock that fills most of the image. It has seen only
crops with a margin of one fifth of the clock's width on each side. A
whole scene with a small clock in it needs a detector or a manual crop
first.

Many photographs cannot be read to the minute by anyone. Four in five
COCO clocks cover under 224 pixels of the source photograph. At that size
one minute of arc is under a pixel wide. On such clocks the model gets
the hour right more often than the minute.

The dial gives no AM or PM, so the hour is always between 1 and 12. A
clock with no minute hand, a digital clock, or a stopped clock all
produce a confident answer that may be wrong. The model was trained and
evaluated on photographs of real clocks. Drawings and rendered clocks
were never tested.

The headline number rests on 200 photographs. A difference of a few
points between two models on this test split is within noise.

## Licence

The base model is released under the LFM Open License v1.0. time-wizard
is a fine-tune of that model and carries the same licence. Read the
licence at the link in the metadata before use.

## Citation

```
@misc{timewizard2026,
  title  = {time-wizard: a 450M model that reads analog clocks},
  author = {Tirmazi, Hayder},
  year   = {2026},
  url    = {https://huggingface.co/jadidbourbaki/time-wizard}
}
```
