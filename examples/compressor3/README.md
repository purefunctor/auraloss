# Dependencies
```bash
# In the project root, run the following
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -e .  # Installs auraloss
$ pip install torchaudio soundfile lightning pandas wandb
```

# Login to Wandb

```bash
$ wandb login
```

# Training

```bash
$ python train.py
```

# Inference

Make sure to change the checkpoint file:

```bash
$ python infer.py
```

# Reporting

Make sure to change the checkpoint file:

```bash
$ python report.py
```
