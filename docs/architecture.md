# Architecture

One image path is shared by training, evaluation and the API: `ml/crowd/transforms.py`. The
API contains no model or normalisation code.

## Training data flow

| Stage | Where | Shape / dtype | Notes |
|---|---|---|---|
| Raw image | `data/ShanghaiTech/.../images/*.jpg` | `(H, W)` or `(H, W, 3)` uint8 | Part A: 189 distinct sizes, 20 grayscale. Part B: all 1024x768. |
| Annotation | `ground-truth/GT_*.mat` | points `(N, 2)`, order `(x, y)` | Clipped to image bounds. |
| Load | `transforms.load_rgb` | `(H, W, 3)` uint8 | Always `.convert("RGB")`. |
| Density map | `data/density.py` -> `data/processed/*.h5` | `(H, W)` float32 | Gaussian per head (geometry-adaptive sigma, fixed fallback). Sums to the head count. |
| Crop + flip | `data/dataset.py` | image `(3, 256, 256)`, density `(256, 256)` | Random crop (multiples of 8), random horizontal flip. |
| Normalise | `transforms.normalize` | `(3, H, W)` float32 | ImageNet mean and std. |
| Target | `transforms.sum_pool` | `(1, H/8, W/8)` | Sum over 8x8 blocks, so the count is preserved exactly. |
| Batch | `DataLoader` | image `(B, 3, 256, 256)`, target `(B, 1, 32, 32)` | Batch size 8 by default. Validation uses batch size 1 on full images. |

## Model (`models/csrnet.py`)

CSRNet: a VGG-16 frontend and a dilated-convolution backend, about 16.3M parameters.

| Stage | Output shape | Detail |
|---|---|---|
| Input | `(B, 3, H, W)` | H and W multiples of 8 |
| Frontend | `(B, 512, H/8, W/8)` | `torchvision.vgg16().features[:23]`, ImageNet-pretrained, 3 max-pools |
| Backend | `(B, 64, H/8, W/8)` | 6 conv layers, dilation 2, channels 512, 512, 512, 256, 128, 64 |
| Output layer | `(B, 1, H/8, W/8)` | 1x1 conv to a density map; its sum is the count |

Loss: `MSELoss(reduction="sum") / batch_size` against the pooled target. Optimiser: Adam, lr 1e-5
(see `config.py` for why not the paper's SGD recipe). `best.pth` is chosen on validation MAE,
where validation is held out of the training split. The test split is not used for selection.

## Part A checkpoint

The shipped Part A model is the weight average of two training runs that started from the same pretrained
network (`python -m crowd.soup`): one plain run and one with random zoom, colour jitter and an
exponential weight average (`--augment --ema 0.995`). It is a single CSRNet, so serving cost is unchanged.

## Serving data flow

```
browser --multipart--> nginx (/api proxy) --> FastAPI --> crowd.predict --> CSRNet
```

| Stage | Where | Shape | Notes |
|---|---|---|---|
| Upload checks | `api/app/uploads.py` | bytes | JPEG/PNG/WebP, at most 10 MB, Pillow decode, pixel limit; 4xx with a readable message |
| Decode | `uploads.decode_image` | `(H, W, 3)` uint8 | EXIF orientation applied |
| Downscale | `transforms.downscale_long_side` | long side at most 1024 | The largest size the models were trained on. Smaller images are not upscaled |
| Normalise + pad | `transforms.normalize`, `pad_to_multiple` | `(1, 3, H', W')`, H' and W' multiples of 8 | Zero padding = mean colour |
| Network | `CSRNet` under `torch.inference_mode()` | `(1, 1, H'/8, W'/8)` | One model per part, loaded once at startup |
| Scene routing | `crowd.routing`, `InferenceService.predict` | | The dense model runs first. Below the threshold in `routing.json` the sparse model takes over |
| Count | `crowd.predict` | scalar | Sum of the output |
| Heatmap | `crowd.viz.overlay_heatmap` | `(H, W, 3)` PNG, base64 | Viridis colour map blended with the photo |
| People per m² | `crowd.area` | scalar | Count inside the region (or the whole photo) divided by the area given |
| Density level | `crowd.area.level_for_density` | Low, Medium, High, Critical | Bands at 1, 2 and 4 people per m² |
| Crowd size | `crowd.levels` | Low, Medium, High, Critical | Count compared with bands from `thresholds.json`, shown when no area is given |

Concurrency: one model instance per part, a semaphore of `MAX_CONCURRENT_INFERENCE` running
requests, a queue of `MAX_QUEUE`, and `503` with `Retry-After` when the queue is full. Uploads stay in memory.

## Density levels

| Level | People per m² |
|---|---|
| Low | under 1 |
| Medium | 1 to 2 |
| High | 2 to 4 |
| Critical | 4 and above |

The bands are in `crowd/area.py`. People per m² is the count inside the marked region divided by the
area the caller gives for it. Without an area the result shows a crowd size (small to very large)
from count quantiles of the ShanghaiTech training annotations (`python -m crowd.calibrate`,
`ml/crowd/thresholds.json`).

## Scene routing

A sparse model undercounts a packed crowd and a dense model overcounts an open street. The dense
model runs first and, when it counts fewer people than the threshold, the sparse model's result is
used. `python -m crowd.routing runs/<dense> runs/<sparse>` fits the threshold on validation images
and writes `ml/crowd/routing.json`.

## Checkpoint safety

Each checkpoint stores the `preprocess` config it was trained with. `load_model` compares it with
the live config and raises on any difference, so the API refuses to start and serves no skewed
predictions. Health returns 503 until every configured model has loaded.
