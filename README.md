# llm-translator

## Installation

```
rye sync
pip install flash-attn --no-build-isolation
pip install "multiprocess==0.70.15"
```

## Dataset Creation

```
bash datasets/download.sh
python src/misc/parse_htm.py
```

## Train

```
accelerate launch --config_file accelerate.json src/train.py
```

## Demo

```
python src/demo.py

# Ja > どうもこんにちは、ニンジャスレイヤーです
# Hello, I'm Ninja Slayer.
```