# Dependencies
```bash
# In the project root, run the following
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -e .  # Installs auraloss
$ pip install torchaudio soundfile lightning pandas
```

# Training

```bash
$ python tcn.py
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
