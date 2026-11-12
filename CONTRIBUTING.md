# Contributing

Thanks for helping out. A few ground rules keep the models honest and the repository small.

1. Set up with `make setup`, and run `make test` and `make lint` before you open a pull request.
2. Keep image preprocessing in `ml/crowd/transforms.py`. Training, evaluation and the API all use it, so there
   is never a second copy that can drift apart.
3. Do not commit the dataset or model files (`.h5`, `.pth`, large images). `pre-commit install` blocks them.
4. Use the test photos only to report a final result. Choose checkpoints, thresholds and settings on the
   validation split.
5. If you change how a model is trained or served, update the matching page in `docs/`:
   [training](docs/training.md), [architecture](docs/architecture.md) or [operations](docs/operations.md).
