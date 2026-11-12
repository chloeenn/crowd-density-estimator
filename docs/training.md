# How the models were trained

This page explains the data, how the two models were trained, how well they work, and how to repeat
the whole process. The exact numbers for each run are in [results.md](results.md) and
[runs/](runs/).

## The data

The models learn from ShanghaiTech (Zhang et al., CVPR 2016), a set of crowd photos where every head
is marked with a dot.

| | Part A | Part B |
|---|---|---|
| Kind of scene | Dense crowds, packed together | Streets and open areas |
| Photos (train / test) | 300 / 182 | 400 / 316 |
| People per photo | 33 to 3,138 (average 541) | 9 to 576 (average 123) |
| Photo sizes | 189 different sizes | all 1024 × 768 |

Two details shape the code. Part A photos come in many sizes, so training uses fixed 256 × 256 crops.
Twenty Part A photos are grayscale, so every image is converted to RGB when it is loaded.

**Density maps.** The dots are turned into a smooth "density map" where each head is a small blob
that adds up to exactly one person. Summing the map gives the count. Part A uses blobs that shrink
in crowded spots (so close-together heads do not smear into each other), and Part B uses a fixed blob
size. `python -m crowd.data.build` creates them once and stores them in `data/processed/`. Six
tests in `ml/tests/test_data_pipeline.py` check that the maps still add up to the right count,
because a mistake here quietly produces a bad model instead of an error.

## The model

Both models are [CSRNet](https://arxiv.org/abs/1802.10062): a VGG-16 front end that starts from
ImageNet-trained weights, plus a dilated-convolution back end that outputs a density map at one
eighth of the photo's size. The estimate is the sum of the map. [architecture.md](architecture.md)
lists the layers and shapes.

## Training recipe

| Setting | Value |
|---|---|
| Loss | Squared error between the predicted and true density maps |
| Optimiser | Adam, learning rate 1e-5, cosine decay |
| Batch | 8 random 256 × 256 crops, randomly flipped |
| Validation | 10% of the training photos, held out with a fixed seed |
| Checkpoint | The epoch with the lowest validation MAE (average error in people per photo) |
| Hardware | Apple M1 (MPS) |

The original paper trains with plain SGD for many more steps. With crops in batches, that setting barely
moved in a 10-photo test, while Adam fit the same photos well, so Adam is the default. The paper's recipe
is still available: `--optimizer sgd --lr 1e-7 --momentum 0.95 --weight-decay 5e-4`.

The test photos are never used to pick checkpoints, thresholds or settings. They are read once per run to
report the final numbers.

## Runs

| Run | Epochs | Best epoch | Validation MAE | Validation RMSE | Time |
|---|---|---|---|---|---|
| `part_A-full` | 200 | 54 | 95.6 | 176.6 | 3.0 h |
| `part_A-aug` | 180 | 168 | 110.9 | 210.7 | 2.0 h |
| `part_A-soup` (ships) | average of the two runs above | | 95.4 | 155.0 | |
| `part_B-full` (ships) | 120 | 107 | 10.5 | 17.5 | 1.9 h |

`part_A-aug` adds random zoom (0.8 to 1.25 times), colour jitter and a running average of the weights
(`--augment --ema 0.995`). Neither Part A run alone was best, but averaging their weights gave one model that
beat both (`python -m crowd.soup`). It works because both runs start from the same pretrained network.
The average is a single model, so serving costs the same.

Things that were tried and not kept:

| Change | Validation MAE (Part A) |
|---|---|
| Keep training the best run with larger crops at a lower learning rate | 110.3 |
| Average the two runs' predictions instead of their weights | 98.6 |
| Run the photo and its mirror image and average | no change |

## Results on the test photos

| Model | Photos | MAE | RMSE | Average error vs truth | Target | Paper |
|---|---|---|---|---|---|---|
| Dense crowds (Part A) | 182 | 85.8 | 122.8 | +21 people (+4.9%) | 85 / 135 | 68.2 / 115.0 |
| Sparse scenes (Part B) | 316 | 11.7 | 18.6 | +2 people | 16 / 24 | 10.6 / 16.0 |

MAE is the average miss in people per photo, and RMSE punishes big misses more. Part B meets its
target. Part A meets the RMSE target and is 0.8 above the MAE target. The scatter plots and the ten worst
photos for each model are in [results.md](results.md).

## Choosing the model for a photo

The sparse model undercounts packed crowds, and the dense model overcounts open streets. So the dense
model runs first, and when it sees fewer people than a threshold (153.5), the sparse model's answer is
used instead. The threshold is fitted on validation photos and capped at the crowd sizes the sparse
model was trained on. On the test photos this lowers the combined relative error from 0.208 to 0.169
(`python -m crowd.routing`, details in `results/routing.json`).

## Density levels

People per m² uses fixed bands:

| Level | People per m² |
|---|---|
| Low | under 1 |
| Medium | 1 to 2 |
| High | 2 to 4 |
| Critical | 4 and above |

When no area is given, the app shows a crowd size instead (small, medium, large, very large). Those bands
come from the 25th, 50th and 75th percentiles of the training photos' counts
(`python -m crowd.calibrate`, stored in `ml/crowd/thresholds.json`).

## Repeat it

```shell
make setup                               # environment and dependencies
make verify-data                         # check the dataset against data/MANIFEST.json
make preprocess                          # build the density maps
python -m crowd.train --part part_A --epochs 200 --lr-schedule cosine --run-id part_A-full
python -m crowd.train --part part_A --epochs 180 --lr-schedule cosine --augment --ema 0.995 --run-id part_A-aug
python -m crowd.train --part part_B --epochs 120 --lr-schedule cosine --run-id part_B-full
python -m crowd.soup runs/part_A-soup runs/part_A-full runs/part_A-aug --val-mae 95.4 --val-rmse 155.0
make eval RUN=runs/part_A-soup           # test photos, read once per run
python -m crowd.routing runs/part_A-soup runs/part_B-full
```

For a faster machine, `ml/notebooks/colab_train.ipynb` runs the same training on a Colab GPU.

Each run writes a folder under `runs/<run_id>/` with its settings, package versions, the train/validation
split, the seed, a per-epoch log, the checkpoints and a summary. A copy of the summary is kept in
`docs/runs/` so the numbers in these docs can be traced to a run. Every checkpoint also stores the
preprocessing it was trained with, and the API refuses to load one that no longer matches the code.
