---
pretty_name: time-wizard-bench
license: other
license_name: mixed
license_details: >-
  Labels and split assignments are MIT. Photographs keep the licence of
  their source. OpenImages photographs are CC BY 2.0. COCO photographs
  carry the Flickr licence chosen by each photographer. See the Licence
  section.
task_categories:
  - image-to-text
  - visual-question-answering
language:
  - en
size_categories:
  - 1K<n<10K
tags:
  - analog-clock
  - clock-reading
  - coco
  - openimages
dataset_info:
  features:
  - name: key
    dtype: string
  - name: image
    dtype: image
  - name: hours
    dtype: int64
  - name: minutes
    dtype: int64
  splits:
  - name: test
    num_bytes: 41056432
    num_examples: 200
  - name: dev
    num_bytes: 42571782
    num_examples: 200
  - name: train
    num_bytes: 561278260
    num_examples: 2796
  download_size: 681512258
  dataset_size: 644906474
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
  - split: dev
    path: data/dev-*
  - split: train
    path: data/train-*
---

# time-wizard-bench

time-wizard-bench holds 3196 photographs of analog clocks. Each photograph
is cropped to one clock and labelled with the hour and minute that clock
shows. The photographs come from COCO and OpenImages. The labels come from
the paper "It's About Time: Analog Clock Reading in the Wild" by Charig
Yang, Weidi Xie, and Andrew Zisserman. The dataset adds fixed test, dev,
and train splits so that any two models can be compared on identical
photographs.

The code that built the dataset, trains on it, and scores against it
lives at
[github.com/jadidbourbaki/time-wizard](https://github.com/jadidbourbaki/time-wizard).

## Fields

| Field | Type | Meaning |
|---|---|---|
| `key` | string | Source and image id, such as `coco_000000009172` |
| `image` | image | The clock crop, 448 by 448 pixels, RGB |
| `hours` | int | Hour on the 12 hour dial, 1 to 12 |
| `minutes` | int | Minute, 0 to 59 |

A key starting with `coco_` names an image in COCO 2017. A key starting
with `openimages_` names an image in OpenImages. The part after the
underscore is the image id in the source dataset. Every crop therefore
traces back to its original photograph.

## Splits

| Split | Photographs | Purpose |
|---|---|---|
| test | 200 | Score a finished model once |
| dev | 200 | Choose hyperparameters and select checkpoints |
| train | 2796 | Fine-tune |

Each split holds COCO and OpenImages photographs in the same proportion
as the whole set, 59 percent COCO. A random draw with seed 0 assigned the
photographs. The split files in the GitHub repository record the
assignment as image ids and labels.

## Crop pipeline

The It's About Time authors found 1911 clocks in COCO photographs and
1317 in OpenImages photographs. The authors read each clock by eye and
recorded the hour and minute. The object detector CBNetV2 drew a box
around each clock. The authors published the labels and the boxes as CSV
files under the MIT licence.

The build script in the GitHub repository turns those files into this
dataset in five steps:

1. Download each photograph from COCO or OpenImages.
2. Crop the photograph to the detector box plus a margin of 20 percent of
   the box on each side. The paper crops with the same margin.
3. Pad the crop to a square with black bars and resize it to 448 pixels.
4. Drop near duplicates. A perceptual hash summarises each crop as a 64
   bit fingerprint. Two crops whose fingerprints differ in four bits or
   fewer show the same clock. The hash check removed 32 photographs.
5. Shuffle each source with seed 0. Assign 200 photographs to test, 200 to
   dev, and the remaining 2796 to train.

## Scoring

Ask the model for the hour and the minute it reads on the clock. A
prediction counts as correct when it lands within one minute of the label
on the 12 hour dial. The distance wraps at twelve o'clock. A prediction of
12:59 against a label of 1:00 is therefore one minute of error.

The It's About Time paper computes its top-1 accuracy the same way. The
paper's numbers on the full COCO and OpenImages sets sit on the same
scale as numbers on this test split.

## Results on the test split

| Model | Parameters | Within 1 min | Hour correct | Mean error |
|---|---|---|---|---|
| time-wizard | 450M | 59.0% | 74.5% | 44 min |
| GPT-5.6 Sol | undisclosed | 60.5% | 68.5% | 59 min |
| Claude Fable 5.1 | undisclosed | 24.0% | 40.0% | 119 min |
| Claude Opus 5 | undisclosed | 15.5% | 31.0% | 141 min |

Every model saw the same 200 crops and the same question. The frontier
models ran at their highest reasoning setting. time-wizard is
LFM2.5-VL-450M after supervised fine-tuning on the train split. The
GitHub repository holds the exact prompt, the scoring code, and every
model's replies.

## Limitations

COCO clocks are small in their source photographs. The median COCO clock
spans 80 pixels. The median OpenImages clock spans 235 pixels. Four in
five COCO crops cover under 224 pixels of the source photograph before
the resize to 448. At that size one minute of arc is under a pixel wide.
No model can read those clocks to the minute.

The labels record hours and minutes only. There is no second hand label,
no AM or PM, and no flag for a clock that shows an impossible time. The
authors read the clocks by eye. Some labels therefore carry a minute or
two of error. Every model is scored against the same labels, so
comparisons between models hold.

The photographs have been public on the internet for years. A model
trained on web data may have seen them along with nearby text. A model
trained on the train split alone has seen only those 2796 crops.

## Licence

The labels, the crop boxes, and the split assignments are MIT. The It's
About Time repository publishes its labels under MIT. time-wizard-bench
keeps that licence for everything derived from them.

The photographs keep the licence of their source. OpenImages photographs
are CC BY 2.0. The `key` field identifies each one for attribution. COCO
photographs come from Flickr under the licence each photographer chose.
COCO distributes them subject to the Flickr terms of use. Use the
photographs on the same terms as their sources.

## Citation

Cite the paper that produced the labels:

```
@inproceedings{yang2022itsabouttime,
  title     = {It's About Time: Analog Clock Reading in the Wild},
  author    = {Yang, Charig and Xie, Weidi and Zisserman, Andrew},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2022}
}
```

Cite the dataset for the splits and the scores:

```
@misc{timewizardbench2026,
  title  = {time-wizard-bench},
  author = {Tirmazi, Hayder},
  year   = {2026},
  url    = {https://huggingface.co/datasets/jadidbourbaki/time-wizard-bench}
}
```
