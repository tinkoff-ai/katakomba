![Katakomba: Tools and Benchmarks for Data-Driven NetHack](katakomba.png)

<p align="center"><b>Katakomba</b> is an open-source benchmark for data-driven NetHack. At the moment, it provides a set of standardized datasets with familiar interfaces and offline RL baselines augmented with recurrence with corresponding logs synced to the Weights&Biases.</p>

## Getting started
TO BE DONE

```bash
git clone https://github.com/tinkoff-ai/katakomba.git && cd katakomba
pip install -r requirements/requirements_dev.txt

# alternatively, you could use docker
docker build -t <image_name> .
docker run --gpus=all -it --rm --name <container_name> <image_name>
```

1. ```cd katakomba/utils/render_utils/```

2. ```pip install -e .```

# Baselines

| Algorithm                                                                                                                       | Variants Implemented                               | Wandb Report |
|---------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------| ----------- |
| ✅ [Behavioral Cloning <br>(BC)](https://www.semanticscholar.org/paper/Cognitive-models-from-subcognitive-skills-Michie-Bain/d40aff59c9b0785e0d75765b0040430ffc377f2d)                                                                                                   | [`bc_chaotic_lstm.py`](algorithms/small_scale/bc_chaotic_lstm.py) |  [`Katakomba-All`]()
| ✅ [Conservative Q-Learning for Offline Reinforcement Learning <br>(CQL)](https://arxiv.org/abs/2006.04779)                      | [`cql_chaotic_lstm.py`](algorithms/small_scale/cql_chaotic_lstm.py)                      | [`Katakomba-All`]()
| ✅ [Accelerating Online Reinforcement Learning with Offline Datasets <br>(AWAC)](https://arxiv.org/abs/2006.09359)               | [`awac_chaotic_lstm.py`](algorithms/small_scale/awac_chaotic_lstm.py)                    | [`Katakomba-All`]()
| ✅ [Offline Reinforcement Learning with Implicit Q-Learning <br>(IQL)](https://arxiv.org/abs/2110.06169)                         | [`iql_chaotic_lstm.py`](algorithms/small_scale/iql_chaotic_lstm.py)                      | [`Katakomba-All`]()
| ✅ [An Optimistic Perspective on Offline Reinforcement Learning <br>(REM)](https://arxiv.org/abs/1907.04543)                     | [`rem_chaotic_lstm.py`](algorithms/small_scale/rem_chaotic_lstm.py)                      | [`Katakomba-All`]()

## Citing Katakamoba
```bibtex

```
