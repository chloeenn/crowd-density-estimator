# Crowd Density Estimator Implementation Plan

## 1. Environment facts

| Item | Value |
|---|---|
| Dataset | ShanghaiTech Part A (300 train, 182 test), Part B (400 train, 316 test) |
| Dataset size | 210 MB |
| Annotations | MATLAB v5 `.mat`. Points at `image_info[0,0][0,0]['location']`, shape `(N,2)`, order `(x, y)` |
| Counts | Part A 33 to 3138 (mean 541). Part B 9 to 576 (mean 123) |
| Image sizes | Part A: 189 distinct sizes. Part B: all 1024x768 |
| Grayscale | 20 Part A images (11 train, 9 test) |
| Model slice | `torchvision.vgg16().features[:23]` gives the first 10 conv layers, 512 channels, stride 8 |
| Hardware | Apple M1, 16 GB RAM, 412 GB free. Use MPS locally and a Colab T4 for training |
| Toolchain | Python 3.11.4, Node 26.7.0, npm 11.19 |
| Installed | `numpy`, `scipy`, `matplotlib`, `Pillow`, `fastapi` |
| Missing | `torch`, `torchvision`, `opencv-python`, `h5py` |
| Git root | `/Users/chloenguyen/Desktop/Project`, shared with other projects |

Handle two dataset facts in code:

1. Part A has 189 image sizes. Use fixed-size random crops to form batches, or use batch size 1.
2. Part A has 20 grayscale images. Call `.convert("RGB")` on every image.

### Git safety

- Stage only paths under `crowd-estimator/`.
- Keep `crowd-estimator/.gitignore` anchored. Use `/data/*` with `!/data/MANIFEST.json`, so `ml/crowd/data/` stays tracked.
- Never run `git add -A`, `git checkout .`, `git stash`, or `git clean` from the parent root.

## 2. Execution rules

1. Inspect before modifying. Read the file, run the test, list the directory.
2. Extend existing code. `ml/crowd/config.py`, `ml/crowd/transforms.py`, `ml/crowd/data/*`, and `ml/tests/test_data_pipeline.py` already exist.
3. Keep the architecture, loss, optimiser, and hyperparameters as documented. Propose a change and update this plan before applying it.
4. Run every test before reporting a result. Include the command and its output.
5. Update this plan when an implementation decision changes.
6. Keep unrelated changes in the parent repo untouched.
7. Meet each phase gate before starting the next phase.
8. Read the test sets only in Phase 4.
9. Keep all serving-time preprocessing in `crowd/transforms.py`. Call it from training, evaluation, and the API.
10. Write metrics, thresholds, and environment details with scripts. Do not type them by hand.

## 3. Phase gates

Stop at each gate and show the evidence.

| Gate | Phase | Condition | Evidence |
|---|---|---|---|
| G0 | 0 | Venv active, dependencies installed, device check passes, dataset manifest verified | `env_check.py` output, `make verify-data` output |
| G1 | 1 | All six data tests pass on real samples | `pytest ml/tests -v`, 6 of 6 green |
| G2 | 2 | Parameter count and output shape asserted | `pytest ml/tests/test_model.py -v` |
| G3 | 3 | 10-image subset overfits to near-zero loss | Final loss value or loss curve |
| G4 | 3 | Full run completes. Best checkpoint selected on validation MAE | `runs/<run_id>/summary.json` |
| G5 | 4 | Test sets evaluated once. Acceptance bar met, or shortfall diagnosed | `docs/results.md` |
| G6 | 5 | Thresholds derived from training data | `thresholds.json` |
| G7 | 6 | API tests pass with a randomly initialised model | `pytest api/tests -v` |
| G8 | 7 | Frontend builds, uploads an image, renders a heatmap | Build output, manual check notes |
| G9 | 8 | Definition of done satisfied | Completed checklist |

If a gate fails, stop and report the failure.

## 4. Targets

Published CSRNet results on ShanghaiTech, for reference only. Quote them with a citation to the paper:

| Part | MAE | RMSE |
|---|---|---|
| A | 68.2 | 115.0 |
| B | 10.6 | 16.0 |

Acceptance bar for this project (Target):

- Part A: MAE <= 85, RMSE <= 135.
- Part B: MAE <= 16, RMSE <= 24.

Expect to reach the bar in 100 to 150 epochs. Paper parity takes about 400 epochs. A Part A MAE above 120 points to wrong density maps. Check Phase 1 first.

Write measured results to `docs/results.md` with `evaluate.py`. Leave this section unchanged after measuring.

## 5. Repository layout

```
crowd-estimator/
├── README.md
├── .gitignore
├── Makefile                   # setup, verify-data, preprocess, train, eval, calibrate, api, web, test
├── requirements-lock.txt      # local pins
├── requirements-colab.txt     # Colab pins, no torch or torchvision
├── docs/
│   ├── PLAN.md
│   ├── architecture.md        # data flow, tensor shapes per stage
│   ├── results.md             # written by evaluate.py
│   ├── results/               # per-part JSON, scatter and worst-10 gallery, written by evaluate.py
│   └── runs/<run_id>.json     # tracked copy of each run summary
├── data/                      # ignored, except MANIFEST.json
│   ├── MANIFEST.json
│   ├── ShanghaiTech/          # raw, read-only
│   └── processed/             # .h5 density maps
├── ml/
│   ├── pyproject.toml         # installable package `crowd`
│   ├── crowd/
│   │   ├── config.py          # frozen dataclasses, YAML-loadable
│   │   ├── env_check.py       # version and device verification
│   │   ├── seeding.py         # set_seed(), RNG capture and restore
│   │   ├── device.py          # cuda, mps or cpu
│   │   ├── schedule.py        # lr_for_epoch(), stateless
│   │   ├── levels.py          # count levels, local concentration, thresholds.json loader
│   │   ├── thresholds.json    # generated by calibrate.py, ships as package data
│   │   ├── metrics.py         # MAE, RMSE, count buckets, acceptance check, diagnosis (pure)
│   │   ├── engine.py          # train_epoch, predict_counts, validate
│   │   ├── checkpoint.py      # atomic save, load_model with preprocess check
│   │   ├── runs.py            # run directories and provenance
│   │   ├── data/
│   │   │   ├── annotations.py # .mat to (N,2) points, bounds clipping
│   │   │   ├── density.py     # geometry-adaptive and fixed-sigma kernels
│   │   │   ├── build.py       # CLI: raw to data/processed/*.h5
│   │   │   └── dataset.py     # crops, flips, 1/8 target, normalisation
│   │   ├── models/csrnet.py   # frontend + dilated backend, about 16.3M params
│   │   ├── transforms.py      # single preprocessing path
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py         # image to (count, density map), used by the API
│   │   ├── calibrate.py       # count thresholds from training data
│   │   ├── viz.py             # OpenCV heatmap overlay (also used by the API)
│   │   └── report.py          # matplotlib plots and results.md renderer (offline only)
│   ├── notebooks/colab_train.ipynb
│   └── tests/
├── runs/                      # ignored, checkpoints and metrics
├── api/
│   ├── requirements.txt       # includes `-e ../ml`; requirements-dev.txt adds test tooling
│   ├── .env.example           # every setting with its default
│   ├── app/
│   │   ├── main.py            # app factory (`uvicorn --factory`), CORS, request id, logging
│   │   ├── config.py          # pydantic-settings
│   │   ├── schemas.py         # response models
│   │   ├── errors.py          # one JSON error shape, no stack traces
│   │   ├── logging_setup.py   # structured JSON logs, whitelisted fields only
│   │   ├── uploads.py         # content-type, size, decode and pixel checks
│   │   ├── deps.py            # settings and model-registry dependencies
│   │   ├── inference.py       # model registry, bounded concurrency, wraps crowd.predict
│   │   └── routers/{predict,system}.py
│   └── tests/
└── web/                       # Vite, React, TypeScript, Tailwind
    └── src/
        ├── api/client.ts
        ├── components/
        ├── hooks/usePrediction.ts
        └── App.tsx
```

## Phase 0: Environment

### Tasks

- [ ] Activate `crowd-estimator/.venv`. Run `python -V` and confirm 3.11.4.
- [ ] Install `torch` first. Install `torchvision` against it so the versions pair.
- [ ] Install the package with `pip install -e "ml[dev]"`.
- [ ] Install `opencv-python` and `h5py`.
- [ ] Write `requirements-lock.txt` with `pip freeze`.
- [ ] Write `requirements-colab.txt` with every package except `torch` and `torchvision`.
- [ ] Write `crowd/env_check.py`.
- [ ] Write `data/MANIFEST.json` and the `make verify-data` target.

### Pinning

- Install the Colab CUDA build of `torch` and `torchvision` from the Colab image. A local wheel replaces it with a CPU build and training slows about 20 times.
- Assert `torch.cuda.is_available()` in the first cell of the Colab notebook.
- Record resolved versions in `env.json` for each run. Leave version numbers out of this plan.

### `env_check.py`

- Print versions of Python, `torch`, `torchvision`, `numpy`, `scipy`, `cv2`, `h5py`.
- Print `torch.backends.mps.is_available()` and `torch.cuda.is_available()`.
- Print the device selected by `config.py`.
- Run a 3x3 conv forward and backward pass on the selected device. Assert the gradient is finite and non-zero.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the local Makefile targets.

### Dataset check

- Count files per part and split.
- Hash the sorted list of relative paths and byte sizes.
- Compare against `data/MANIFEST.json`. Fail on any difference.

Gate G0: `env_check.py` and `make verify-data` pass.

## Phase 1: Data pipeline

Density map errors raise no exceptions. They produce a model with a poor MAE. Test this phase before training.

### Annotations (`annotations.py`)

- [ ] Parse the nested array. Return `float32` shape `(N,2)`.
- [ ] Clip points to image bounds. Some points fall outside the frame by a fraction of a pixel.

### Density maps (`density.py`)

Convert each head to a normalised Gaussian so `density_map.sum() == len(points)`.

- Part A: geometry-adaptive kernel. Use `sigma_i = beta * d_i`, where `d_i` is the mean distance to the `k` nearest neighbours. Set `beta = 0.3` and `k = 3`. Use `scipy.spatial.KDTree`.
- Part B: fixed `sigma = 15`.
- `N == 0`: return a zero map.
- `N == 1`: use the fixed sigma.
- Store `beta`, `k`, and sigma in `config.py`.

### Precompute (`build.py`)

- [ ] Write one `.h5` per image to `data/processed/`. Mirror the raw directory tree.
- [ ] Run once. This produces about 1200 files in a few minutes.

### Dataset (`dataset.py`)

Apply these steps to each item in order:

1. Load the image and call `.convert("RGB")`.
2. Load the `.h5` density map.
3. Training only: random crop to a fixed size (256x256 or one quarter of the image). Apply a random horizontal flip to the image and the map together.
4. Downsample the target to 1/8 and restore the count:
   `cv2.resize(t, (w//8, h//8), interpolation=cv2.INTER_CUBIC) * 64`.
   Skipping the `* 64` shrinks targets about 64 times and the model learns to output near zero.
5. Normalise with the ImageNet mean and standard deviation.

### Tests (`ml/tests/test_data_pipeline.py`)

| Test | Checks |
|---|---|
| Density sum equals point count within 1% on 20 real samples | Map normalisation |
| Count after the 1/8 downsample within 2% | The `* 64` factor |
| Grayscale image returns shape `(3, H, W)` | RGB conversion |
| Out-of-bounds points are clipped and kept | Count preservation |
| Flip changes image and map identically | Supervision alignment |
| `N == 0` and `N == 1` return finite maps | Kernel edge cases |

Gate G1: all six tests pass on real dataset samples. Show the `pytest` output.

## Phase 2: Model (`models/csrnet.py`)

- [ ] Build the frontend: `vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23]`.
- [ ] Build the backend: 6 conv layers, kernel 3, `dilation=2`, `padding=2`, channels `512, 512, 512, 256, 128, 64`, ReLU after each.
- [ ] Add the head: `Conv2d(64, 1, kernel_size=1)`.
- [ ] Initialise the backend with `normal(0, 0.01)` and biases 0. Keep pretrained weights in the frontend.
- [ ] Pad H and W to multiples of 8 at inference.
- [ ] Write `ml/tests/test_model.py`. Assert a parameter count near 16.3M. Assert input `(1,3,256,256)` returns output `(1,1,32,32)`.

Gate G2: `test_model.py` passes.

## Phase 3: Training (`train.py`)

### Recipe

| Setting | Value |
|---|---|
| Loss | `MSELoss(reduction="sum")` divided by batch size |
| Optimiser | Adam, lr `1e-5`, no weight decay (default). Paper recipe: SGD, lr `1e-7`, momentum 0.95, weight decay `5e-4` |
| Batch | 1 for full images, 8 to 16 for fixed crops |
| Epochs | 100 to 150 for the acceptance bar, about 400 for paper parity |
| Checkpoints | `best.pth` on validation MAE, `last.pth` every epoch |

Loss reduction and learning rate depend on each other. `reduction="sum"` adds over about 12k output pixels, so gradients run about 10^4 times larger than with `reduction="mean"`. Change both together. Document this next to the values in `config.py`. Phase 3 findings below show why Adam at `1e-5` is the default.

### Phase 3 findings (measured 2026-11-09)

#### Optimiser choice

The paper's SGD recipe needs single full images and 100k+ steps. This loop uses fixed-size crops in batches of 8, so it takes far fewer steps. Adam is the default. Overfit check on 10 Part B images (predict-nothing baseline MAE 133.8):

| Optimiser | Steps | Val MAE | Note |
|---|---|---|---|
| SGD lr 1e-7 (paper) | 150 | 90.5 | Still descending. Too slow for this step count |
| SGD lr 1e-6 | 200 | 56.2 | Plateau from step 60 |
| Adam lr 1e-5 | 150 | 23.8 best | Fits. Density correlation with ground truth 0.86 |
| Adam lr 1e-5 | 600 | 10.3 best, 10.6 final | Gate G3 run |

Defaults live in `TrainConfig`. Run the paper recipe with `--optimizer sgd --lr 1e-7 --momentum 0.95 --weight-decay 5e-4`. The `reduction="sum"` pairing applies to both optimisers.

#### G3: overfit 10 images (passed, one open issue)

- One fixed batch: loss fell from 1.34 to 0.0007. Predicted count 10.6, target 10.5.
- 10-image run: train loss fell from 1.3 to 0.03-0.1 per sample. Validation MAE on the same images reached 10.3 (8% of the mean count, 134).
- Open issue: validation MAE oscillates between 10 and 38 across checkpoints at a constant learning rate. Crop loss is near zero. Full-image counts are unstable.
- Decision on a learning-rate schedule: **keep `constant` as the default.** A stateless cosine schedule (`--lr-schedule cosine`, `crowd/schedule.py`) is implemented and tested, but the same 600-step overfit run showed no benefit:

| Schedule | Best val MAE | Final val MAE | Note |
|---|---|---|---|
| constant 1e-5 | 10.3 (epoch 449) | 10.6 | oscillates 10 to 38 between checkpoints |
| cosine 1e-5 to 1e-7 | 11.5 (epoch 599) | 11.5 | still oscillates 12 to 27 while lr is near zero |

- The oscillation therefore does not come from learning-rate noise, and its cause is not yet understood. Best-checkpoint selection on validation MAE already guards against it. Re-test the schedule on the held-out validation curve of the full run (G4), where it can be measured on unseen images.

#### G4: full run (open)

G4 needs Colab GPU time. The Part A and Part B acceptance numbers in Section 4 are not validated.

### Tasks

- [x] Train the full network. Add `--freeze-frontend-epochs` with default 0 for an optional warmup of 1 to 2 epochs. A permanently frozen frontend costs 10 to 20 MAE on Part A.
- [x] Hold out 10% of the training set as validation. Use a recorded seed. Save the split to `split.json`.
- [x] Select checkpoints on validation MAE.
- [x] Implement `--resume`. Restore model, optimiser, epoch, best MAE, and RNG state. Append to the same `metrics.jsonl`.
- [x] Write a run directory for every run (see Versioning).
- [ ] Train Part A and Part B as separate models.
- [x] Write `notebooks/colab_train.ipynb`. Save a checkpoint to Drive every epoch. Upload the dataset to Drive once.

### Compute

- Colab T4, Part A: 1.5 to 2.5 min per epoch at batch 1, about 3 to 5 h for 150 epochs.
- Colab T4, Part B: faster with crop batching.
- M1 MPS: 3 to 5 times slower than the T4. Use it for smoke tests and the overfit run.

### Gates

- G3: overfit 10 images to near-zero loss before any full run. On failure, debug Phase 1.
- G4: complete the full run. Confirm `summary.json` exists and the selected checkpoint has the best validation MAE.

## Phase 4: Evaluation (`evaluate.py`)

Run this phase once per part on the final checkpoint. This is the first read of the test sets.

- [x] Compute MAE and RMSE per part.
- [x] Break results down by ground-truth count: 0 to 100, 100 to 500, 500 to 1000, 1000 and above.
- [x] Plot predicted against true counts with matplotlib.
- [x] Build a gallery of the 10 worst images: image, ground truth, prediction, heatmap.
- [x] Write all results to `docs/results.md` and update the run `summary.json`.
- [x] If the acceptance bar is missed, report the measured numbers and a diagnosis. Keep the same checkpoint.

Gate G5: `docs/results.md` regenerated by the script.

#### Phase 4 status: tooling done, G5 open

The checked items above are implemented and tested. Gate G5 itself stays open, because it needs the final Part A and Part B checkpoints from G4, and the test sets are read once per run.

- `python -m crowd.evaluate runs/<run_id>` evaluates `best.pth` on the test split. It writes `docs/results/<part>.json`, the scatter plot and the worst-10 gallery, regenerates `docs/results.md`, records the numbers in the run's `summary.json`, and re-publishes `docs/runs/<run_id>.json`.
- A second test evaluation of the same run is refused unless `--force` is passed. `--split val` is a dry run on the run's validation images and writes only inside the run directory.
- Verification without reading any test data: 13 end-to-end tests on a synthetic ShanghaiTech layout, and a dry run on the real 10-image overfit checkpoint (validation split). That run reproduced the training-time validation metrics exactly (MAE 10.2927, RMSE 17.6227).
- Error analysis in `results.md` is generated from the data: mean bias, the count bucket that carries most of the error, and the share from the 10 worst images. A Part A MAE above 120 also prints the Phase 1 hint from Section 4.


## Phase 5: Density levels (`calibrate.py`)

Head count describes crowd size. Framing changes how dense a scene looks. Report both a count level and an image-space concentration value.

### Count thresholds

- [x] Pool the Part A and Part B training splits. Exclude validation images.
- [x] Compute the 25th, 50th, 75th, and 90th percentiles of the counts.
- [x] Use them as the Low, Medium, High, and Critical band edges.
- [x] Write `thresholds.json` with the band values, quantiles, split identifier, image count, parts pooled, validation seed, and git SHA.
- [ ] Document in the README: the bands reflect ShanghaiTech counts. They are a default for demo use.

### `local_concentration`

- [x] Compute the 95th percentile of per-cell density from the predicted map. Each cell covers 8x8 input pixels at the processed scale.
- [ ] Document it as an image-space estimate. Camera distance, resolution, cropping, and perspective all change it. It does not give people per square metre.
- [x] Return `concentration_basis` with the cell size and processed dimensions. Compare values only between images at a similar scale.
- [ ] Show the framing-dependence caveat next to the level in the UI.

Gate G6: `thresholds.json` records a training split.

#### Phase 5 status: calibration done (pulled forward for Phase 6), README items open

Phase 6 returns `level`, so the count bands and `local_concentration` were built first. They need no trained model: the bands come from training annotations only (`python -m crowd.calibrate`, test split never read).

- **G6 evidence:** `ml/crowd/thresholds.json` records `split: "train_data minus validation"`, 630 images (700 training minus 70 validation), parts pooled A and B, validation seed 0, git SHA. Independently re-derived with NumPy straight from the `.mat` files: p25/50/75/90 = 85 / 166 / 350.75 / 687.2.
- **Decision, level rule.** Four percentiles cannot define four bands with three cut points, so each edge is where its band *starts*: Critical from 687, High from 351, Medium from 166, and Low below 166. `low` (85) is stored and returned as the lowest quantile but does not change a level. Bands are therefore Low about 50%, Medium 25%, High 15%, Critical 10% of ShanghaiTech training images.
- **Decision, location.** The file ships as package data at `ml/crowd/thresholds.json`. Its SHA-256 is written into every new checkpoint (`thresholds_sha`); the API flags a model whose SHA differs (`/api/metadata` `thresholds_match`) and logs a warning.
- `calibrate.py` splits plain image paths with the same `train_val_split` the trainer uses; a test asserts the excluded validation images are identical.
- Still open: the README note that bands reflect ShanghaiTech counts (Phase 8), and the framing caveat in the UI (Phase 7). The caveat text is served by `/api/metadata` as `level_caveat`.


## Phase 6: API (FastAPI)

### Endpoints

| Route | Purpose |
|---|---|
| `POST /api/predict` | Multipart image in, prediction JSON out |
| `GET /api/health` | Liveness and model-loaded status |
| `GET /api/metadata` | Model version, part, thresholds, limits |

Response:

```json
{
  "count": 412.7,
  "level": "High",
  "local_concentration": 1.83,
  "max_concentration": 2.41,
  "concentration_basis": {
    "cell_input_px": 8,
    "processed_width": 1600,
    "processed_height": 1067,
    "units": "people per 8x8-input-pixel cell (image-space, not people/m^2)"
  },
  "thresholds": {"low": 50, "medium": 150, "high": 400},
  "model": {"version": "csrnet-partA-v1", "part": "A", "git_sha": "abc1234"},
  "heatmap": "data:image/png;base64,...",
  "inference_ms": 240
}
```

Return the heatmap as inline base64 in v1.

### Tasks

- [x] Load the model once in the lifespan handler.
- [x] Run inference under `torch.inference_mode()`. Call `torch.set_num_threads(1)`.
- [x] Call `crowd.predict` for preprocessing. Add no normalisation code to `api/`.
- [x] Compare the checkpoint `preprocess` config with the live config at load time. Refuse to start on a mismatch.
- [x] Validate uploads:
  - Allow JPEG, PNG, and WebP content types.
  - Limit size to 10 MB.
  - Decode with Pillow. Set `MAX_IMAGE_PIXELS`.
  - Verify the file is a real image.
  - Downscale the long side to 1600 px or less.
  - Pad to a multiple of 8.
  - Return 4xx with a readable message. Never return a stack trace.
- [x] Render the heatmap with OpenCV: `applyColorMap(norm, COLORMAP_JET)`, upsample to the original size, then `addWeighted(img, 0.6, heat, 0.4)`.
- [x] Normalise each heatmap per image. Return `max_concentration` for the scale label. Accept an optional fixed scale parameter for comparable heatmaps across images.
- [x] Serve Part A by default. Add a model switch for Part B.
- [x] Write `api/tests/` with a randomly initialised model:
  - Valid image returns 200 and a well-formed body.
  - 15 MB file returns 413.
  - `.txt` file returns 415.
  - Truncated JPEG returns 400.

Gate G7: `pytest api/tests -v` passes.

#### Phase 6 status: done, G7 met

- **Evidence:** 59 API tests pass with randomly initialised checkpoints that go through the real load path (including the preprocess check). The four cases the plan names are covered (valid image 200, 15 MB file 413, `.txt` 415, truncated JPEG 400), plus the model switch, fixed heatmap scale, EXIF orientation, startup refusal, CORS, request ids, log hygiene and admission control.
- **Real-server smoke test** (uvicorn, the 10-image Part B checkpoint, a real Part B image): HTTP 200 with count 191.1, identical to the evaluator's prediction for that image (parity over HTTP). 415, 400, 413 and 422 returned clean JSON. Nine simultaneous requests: 5 served (1 running, 4 queued) and 4 got `503` with `Retry-After: 2`.
- **Latency:** about 3.5 s for a 1024x768 image on one CPU thread (Apple M1, `TORCH_THREADS=1`). Requests run one at a time, so a queue of 5 takes about 17 s end to end. Raise `TORCH_THREADS` or `MAX_CONCURRENT_INFERENCE` to trade memory for speed.
- **Parity:** `crowd.predict` shares `crowd.transforms` with training and evaluation; a test asserts its counts equal the evaluation path's, and another that the API count equals the library count for the same bytes. `api/` contains no normalisation or model code.

Decisions and deviations from the text above:

- **Run command:** `uvicorn --factory app.main:create_app --app-dir api`. No module-level app, so importing has no side effects.
- **Health:** models load inside the lifespan before the server accepts connections, so a bad checkpoint makes startup fail (the plan's "refuse to start"). `GET /api/health` returns 503 whenever models are not loaded (for example an app object whose lifespan has not run). Before startup completes the port is closed, which fails a Docker healthcheck the same way.
- **Heatmap size:** the overlay is rendered at the processed (downscaled, at most `MAX_LONG_SIDE`) size, not the uploaded size, to bound the base64 payload. `concentration_basis` reports those dimensions.
- **Pixel limit:** `MAX_IMAGE_PIXELS` is enforced per app inside `decode_image` and returns 413. The API does not assign Pillow's process-global `Image.MAX_IMAGE_PIXELS` (the last app created would win); Pillow's defaults stay as a second line of defence and map to the same 413.
- **Uploads stay in memory.** Starlette spools multipart files over 1 MB to a temp file on disk. The app raises `MultiPartParser.spool_max_size` above the upload limit (a test proves a 2 MB upload stays in memory, and a control test proves the default would have written to disk). The setting is process-wide, so it only ever increases.
- **Extra settings** beyond the plan's list, all in `.env.example` and tested against the defaults: `ALT_MODEL_PATH` (the second model for the switch), `MAX_IMAGE_PIXELS`, `DEVICE`, `TORCH_THREADS`, `MAX_CONCURRENT_INFERENCE`, `MAX_QUEUE`, `RETRY_AFTER_S`.
- **Query parameters:** `part=A|B` selects the model and `scale_max` (people per cell) fixes the heatmap scale.
- **Errors:** every failure is `{"detail": <message>, "code": <slug>}`. Unexpected errors return a generic 500 and are logged server-side with the request id.
- **Open follow-ups:** Dockerfile, CI and Makefile targets (DevOps section). `api/requirements.txt` pins direct dependencies only until the lock file exists. Starlette's TestClient prints a deprecation warning that it prefers a newer `httpx` fork; I did not install an unverified package for it and filtered that one warning in `pytest.ini`.


## Phase 7: Frontend (Vite, React, TypeScript, Tailwind, Axios)

### Components

- [x] `Dropzone`: drag and drop, file picker, paste, keyboard access.
- [x] `HeatmapViewer`: heatmap over the original, opacity slider, A/B toggle.
- [x] `ResultCard`: count, level badge, local concentration, inference time, caveat line.
- [x] `ScaleLegend`: blue to red ramp labelled from `max_concentration`, with the image-space note.
- [x] `SampleGallery`: 3 to 4 bundled test-set images.
- [x] `ErrorBanner`: plain-language messages for 413, 415, 400, and 503.
- [x] Model switch for Part A and Part B, read from `/api/metadata`.

### Behaviour

- [x] Implement the state machine `idle`, `uploading`, `inferring`, `done`, `error`.
- [x] Cancel a superseded request with `AbortController`.
- [x] Show the level with an icon and text. Avoid colour as the only signal.
- [x] Add alt text to every image.
- [x] Read the API base URL from `VITE_API_URL`.

Gate G8: build the app, upload an image against the real API, confirm the heatmap renders.

#### Phase 7 status: built, G8 met through the dev proxy (no browser run)

- **Evidence:** `npm run build`, `npm run lint` and `npm test` (15 tests) pass. Against the real API (uvicorn, the 10-image Part B checkpoint, Vite dev server proxying `/api`): `/api/metadata` and `POST /api/predict` on a bundled sample returned 200 with count 31.1, level Low and a valid `data:image/png;base64` heatmap; I decoded it and it shows the overlay on the right people. A `.txt` upload returned a clean 415 through the proxy.
- **Not verified:** the page was never opened in a real browser, so drag and drop, paste, the opacity slider, the A/B toggle, and focus rings are covered only by the component tests and by reading the code. Open `npm run dev` once and click through before treating G8 as fully met.
- **Decisions:** Tailwind v4 via `@tailwindcss/vite`. `VITE_API_URL` is empty by default, so the client calls same-origin `/api` (Vite proxies it in dev; nginx will in Docker, so no CORS). The returned heatmap is already blended with the photo, so "heatmap over the original" is the overlay stacked on the original with an opacity slider, and the A/B toggle switches between Heatmap and Original. The model switch is hidden when `/api/metadata` lists only one model, and changing it re-runs the current image. Four bundled samples (two Part A, two Part B test images, under 250 KB each) are displayed and run through the model for the demo only; they are not used for selection or calibration. Their ShanghaiTech licence terms need a README note (Phase 8).
- **Levels:** the API returns four levels (Low, Medium, High, Critical); each has its own icon and text label.
- **Open:** `web/Dockerfile` with nginx proxying `/api`, and the CI `web` job (DevOps section).


## Phase 8: Documentation and polish

- [ ] Write `README.md`: description, upload GIF, 3-command quickstart, results table, architecture diagram, Limitations.
- [ ] List these limitations: trained on ShanghaiTech street and plaza scenes. Accuracy drops on aerial, thermal, night, and heavily occluded images. The level label depends on framing. `local_concentration` is image-space. The tool is unsuitable for safety-critical monitoring.
- [ ] Write `docs/architecture.md` with tensor shapes per stage.
- [ ] Complete the Makefile targets: `setup`, `verify-data`, `preprocess`, `train`, `eval`, `calibrate`, `api`, `web`, `test`.
- [ ] Configure `ruff` (lint and format) with pre-commit. Add type hints to public functions. Write docstrings with the reason for each function.
- [ ] Add `make download-weights`. Verify a SHA-256. Publish the ~62 MB checkpoints as a GitHub release asset or Drive link.
- [ ] Add Docker, CI, and runtime configuration as specified in "DevOps and delivery" below.

Gate G9: complete the Definition of done.

## DevOps and delivery

Scope is local deployment plus CI. No cloud hosting, Kubernetes, or GPU serving. Build this alongside Phases 6 to 8. CI steps start after Phase 1.

### CI (GitHub Actions)

The git root is the shared parent. Put the workflow in the parent `.github/workflows/crowd-estimator.yml` and scope it with `paths: ["crowd-estimator/**"]` and `defaults.run.working-directory: crowd-estimator`. Do not touch other projects' workflows.

- [ ] Job `ml`: Python 3.11, install CPU `torch` and `torchvision` from the PyTorch CPU index, `pip install -e "ml[dev]"`, run `ruff check`, `ruff format --check`, and `pytest ml/tests`.
- [ ] Job `api`: install `api/requirements.txt`, run `pytest api/tests` with a randomly initialised model.
- [ ] Job `web`: `npm ci`, `npm run lint`, `npm run build`.
- [ ] Job `docker`: build both images without pushing, to catch Dockerfile drift.
- [ ] Cache pip and npm. Cancel superseded runs with `concurrency`.
- [ ] CI has no dataset. Mark tests with real samples as `@pytest.mark.dataset` and skip them in CI. G1 evidence still comes from the local run. Add synthetic-data variants of the six data tests so CI covers the logic.
- [ ] CI never downloads the 62 MB checkpoints. API tests use a random model.
- [ ] Add Dependabot for pip, npm, and Docker, scoped to `crowd-estimator/`.

### Containers

- [ ] `api/Dockerfile`: `python:3.11-slim`, CPU `torch` wheels, non-root user, `HEALTHCHECK` on `/api/health`. Build context is the repo root of `crowd-estimator/` because the API installs `-e ../ml`. Add `.dockerignore` for `data/`, `runs/`, `.venv/`, `web/node_modules/`, `.git`, `.history`.
- [ ] Do not bake checkpoints into the image. Mount `./models:/models:ro` and set `MODEL_PATH`. The API fails fast if the checkpoint is missing or the `preprocess` check fails, so the container exits non-zero and the compose status shows it.
- [ ] `web/Dockerfile`: multi-stage. Node builds, nginx serves the static bundle. `VITE_API_URL` is a build-time value, so either proxy `/api` through nginx (preferred, no CORS) or document the rebuild.
- [ ] `docker-compose.yml`: services `api` and `web`, a `models` volume, `depends_on` with `condition: service_healthy`, and a restart policy.
- [ ] Image size check: CPU torch image should stay under about 1.5 GB. Record the size in the README.
- [ ] Set `torch.set_num_threads` from an env var with default 1 (see Phase 6), and set a container memory limit sized for a 1600 px input.

### Runtime configuration and operations

- [ ] All API settings come from environment variables through `pydantic-settings`: `MODEL_PATH`, `MODEL_PART`, `CORS_ORIGINS`, `MAX_UPLOAD_MB`, `MAX_LONG_SIDE`, `LOG_LEVEL`. Commit `.env.example`. Ignore `.env`.
- [ ] Restrict CORS to the configured origins. No `*` in the compose file.
- [ ] Structured JSON logs with request id, status, and `inference_ms`. Never log image bytes or filenames.
- [ ] `GET /api/health` returns 503 until the model is loaded, so the Docker healthcheck is meaningful.
- [ ] Serialise or bound concurrent inference. One model instance on CPU with a semaphore, and 503 with `Retry-After` when the queue is full.
- [ ] Uploads are processed in memory and never written to disk. Say so in the README privacy note.
- [ ] Pin Python dependency versions in `api/requirements.txt` from the lock file, so the image matches CI. Align `ml/pyproject.toml` lower bounds (`torch>=2.2`) with what CI and Docker install.

### Model artefact distribution

- [ ] Publish `best.pth` for each part as a GitHub release asset with a SHA-256 file. Tag releases `v<major>.<minor>` and keep the tag, `model_version`, and `docs/runs/<run_id>.json` in step.
- [ ] `make download-weights` verifies the hash and writes to `./models/`. Compose and the README quickstart call it.
- [ ] Training stays on Colab with Drive checkpoints. No training in CI or Docker.

### Developer tooling

- [ ] Use `ruff` for both linting and formatting, as configured in `ml/pyproject.toml`. Do not add `black`.
- [ ] Add `.pre-commit-config.yaml` with `ruff`, `ruff-format`, a large-file check (blocks `.h5`, `.pth`, images over 1 MB), and end-of-file fixes. The large-file hook backs up the Git safety rule against committing the dataset.
- [ ] Makefile targets `docker-build`, `docker-up`, `docker-down`, and `lint`. Keep the `PYTORCH_ENABLE_MPS_FALLBACK=1` setting out of Docker and CI.

### Out of scope

Cloud deploy, autoscaling, GPU inference, metrics and tracing stack, authentication, rate limiting beyond the queue bound, and model registries.

## Versioning

### Dataset

Track `data/MANIFEST.json`:

```json
{
  "name": "ShanghaiTech Part A + Part B",
  "source": "Download URL or archive name of this copy",
  "obtained": "2026-11-09",
  "parts": {
    "A": {"train": 300, "test": 182},
    "B": {"train": 400, "test": 316}
  },
  "filelist_sha256": "<hash of sorted relative paths and byte sizes>",
  "size_bytes": 220200960
}
```

### Training run

Set `run_id = <part>-<YYYYMMDD-HHMM>-<git_sha>`. Write `runs/<run_id>/`:

| File | Contents |
|---|---|
| `config.yaml` | Resolved config with all defaults |
| `env.json` | Package versions, device, platform |
| `dataset.json` | Manifest hash used for training |
| `split.json` | Train and validation image lists, split seed |
| `seed.json` | Global seed and seeded RNG backends |
| `metrics.jsonl` | Per epoch: train loss, val MAE, val RMSE, lr, wall time |
| `best.pth`, `last.pth` | Checkpoints |
| `summary.json` | Best epoch, best val MAE and RMSE, total time, test metrics after Phase 4 |

Copy `summary.json`, `config.yaml`, and `env.json` into `docs/runs/<run_id>.json`. `runs/` is ignored and `docs/runs/` is tracked.

### Checkpoint metadata

Store this dict next to the weights in every `.pth`:

```
model_version        "csrnet-partA-v1"
part                 "A" or "B"
run_id, git_sha, seed, epoch
val_mae, val_rmse
preprocess           {mean, std, stride, pad_multiple, sigma_strategy, beta, k, fixed_sigma}
dataset_manifest_sha, thresholds_sha
```

`crowd.predict` compares `preprocess` with the live config and raises on any difference.

### Seeding

- [ ] Implement `crowd/seeding.py:set_seed(seed)`. Seed `random`, `numpy`, and `torch` on all devices. Save the value in the run directory.
- [ ] cuDNN and MPS kernels are nondeterministic. Reproduce the split and initialisation from the seed.

## Schedule

| Phase | Work | Duration |
|---|---|---|
| 0 | Environment, pins, device check, manifest | 2 h |
| 1 | Data pipeline and six tests | 1 day |
| 2 | Model and tests | 2 h |
| 3 | Training loop, Colab notebook, overfit run | 0.5 day, plus 3 to 5 h GPU per part |
| 4 | Evaluation and reporting | 3 h |
| 5 | Threshold calibration | 2 h |
| 6 | API and tests | 0.5 day |
| 7 | Frontend | 1 day |
| 8 | Docs, Makefile, polish | 0.5 day |

Total: 5 to 7 working days plus unattended training. Build Phases 6 and 7 against a randomly initialised checkpoint while training runs.

## Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Density map sum differs from head count | Critical | Phase 1 test, G1 |
| 2 | Missing `* 64` after the 1/8 downsample | Critical | Phase 1 test, G1 |
| 3 | Dataset committed to the shared parent repo | High | Anchored `.gitignore`, staging rule in Section 1 |
| 4 | 20 grayscale Part A images | High | `.convert("RGB")` and test |
| 5 | Train and serve preprocessing differ | High | Single `crowd.predict`, checkpoint preprocess check |
| 6 | Test data leaks into calibration or checkpoint selection | High | Calibrate on train, select on validation, G4 and G6 |
| 7 | Local torch pin overwrites Colab CUDA build | High | `requirements-colab.txt`, CUDA assert in cell 1 |
| 8 | Run provenance lost with `runs/` ignored | Medium | Tracked `docs/runs/` copy |
| 9 | Colab disconnect loses a run | Medium | Per-epoch Drive checkpoint, `--resume` |
| 10 | Learning rate and loss reduction mismatch | Medium | Documented pairing in `config.py` |
| 11 | Overfitting on 300 Part A images | Medium | Crops, flips, early stop on validation MAE |
| 12 | MPS returns NaN from an accepted graph | Medium | Backward check in `env_check.py` |
| 13 | `local_concentration` read as people per m^2 | Medium | Rename, caveats in API, UI, README |
| 14 | Points outside image bounds | Low | Clip in `annotations.py` |
| 15 | 189 Part A sizes block batching | Low | Fixed-size crops or batch size 1 |
| 16 | CI workflow in the shared parent repo triggers on or breaks other projects | Medium | `paths` filter, `working-directory`, no edits to other workflows |
| 17 | Docker image bloated by CUDA torch or baked-in weights | Medium | CPU wheels, mount checkpoints, size check |
| 18 | Container starts with missing or mismatched checkpoint | Medium | Fail fast at load, health returns 503, compose healthcheck |
| 19 | Dataset-dependent tests fail in CI, or get disabled and hide regressions | Medium | `dataset` marker, synthetic variants run in CI |
| 20 | `VITE_API_URL` baked at build time breaks in compose | Low | nginx `/api` proxy |

## Definition of done

### Correctness

- [ ] Six data tests pass on real samples (G1).
- [ ] Model parameter count and output shape asserted (G2).
- [ ] 10-image overfit reaches near-zero loss (G3).
- [ ] Part A and Part B trained, best checkpoints chosen on validation (G4).
- [ ] Test sets evaluated once, `docs/results.md` generated by script (G5).
- [ ] Acceptance bar met, or shortfall diagnosed in `docs/results.md`.
- [ ] `thresholds.json` derived from training data (G6).

### Software

- [ ] `make test` passes for ml and api.
- [ ] `ruff check` and `ruff format --check` report no issues.
- [ ] CI passes on a pull request for `ml`, `api`, `web`, and `docker` jobs.
- [ ] `docker compose up` starts both services healthy against downloaded weights, and an upload works through the browser.
- [ ] `.env.example` lists every setting. No secrets or `*` CORS in committed config.
- [ ] API returns a valid body for a valid image and the correct code for each of the four bad inputs (G7).
- [ ] A deliberate preprocessing mismatch raises at checkpoint load.
- [ ] Frontend builds and completes an upload against the real API (G8).
- [ ] No preprocessing code exists outside `crowd/transforms.py`.

### Reproducibility

- [ ] `requirements-lock.txt` and `requirements-colab.txt` committed.
- [ ] `env_check.py` passes on macOS with MPS and on Colab with CUDA.
- [ ] `data/MANIFEST.json` committed. `make verify-data` passes.
- [ ] `docs/runs/<run_id>.json` committed for every run with quoted numbers.
- [ ] Every quoted metric links to a run.

### Documentation

- [ ] README complete with Limitations.
- [ ] `docs/architecture.md` lists tensor shapes per stage.
- [ ] This plan matches the built system.

### Hygiene

- [ ] `git status` from the parent shows no staged paths outside `crowd-estimator/`.
- [ ] `git ls-files crowd-estimator | grep -E '\.(h5|pth|pt|jpg)$'` returns nothing.
- [ ] Other projects in the parent repo are unchanged.

## Handoff report

Write this report when the Definition of done is complete:

1. Files changed: added, modified, deleted, grouped by phase.
2. Commands run: setup, preprocessing, training, evaluation, tests, in order.
3. Test results: raw output and counts for each suite.
4. Model performance: results table with a link to each `docs/runs/<run_id>.json` and the acceptance bar.
5. Deviations from this plan, with the reason for each.
6. Known limitations: measured failure modes from the worst-10 gallery, plus the README limitations.
7. Remaining work with enough detail to resume.
8. Gate evidence: G0 to G9, each with its artifact or output.

## Decisions

- Train Part A and Part B separately. Serve Part A by default with a UI switch. A combined model is out of scope.
- Target local deployment. Provide Docker files for `api` and `web`, plus GitHub Actions CI. No cloud hosting.
- Use `ruff` for lint and format. Drop `black`.
- Ship model weights as release assets, mounted into the container, never baked into images or committed.
- Use the 25/50/75/90 percentile bands for Low, Medium, High, and Critical.
- Keep `data/MANIFEST.json` tracked in `data/`. Keep run summaries in `docs/runs/`.
