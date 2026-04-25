# Inference-Time Alignment with Differential Privacy

This repository contains reference implementations of four inference-time alignment algorithms for large language models, including two with formal **differential-privacy (DP)** guarantees:

| Algorithm | File | Privacy | Reference |
|---|---|---|---|
| **BoN** — Best-of-N | `bon.py` | None | Stiennon et al., 2020 |
| **PrivBoN-Gumbel** — Private Best-of-N via Exponential Mechanism | `privbon_gumbel.py` | (ε, 0)-DP | Algorithm 2 in our paper |
| **ITP** — InferenceTimePessimism | `itp.py` | None | Huang et al., 2025 |
| **PrivITP** — Private InferenceTimePessimism (two-phase) | `privitp.py` | (ε, δ)-DP | Algorithm 1 in our paper |

All four algorithms work with any HuggingFace causal LM as the base policy and any HuggingFace sequence-classification model as the reward model.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithm Details](#algorithm-details)
- [Hyperparameter Guide](#hyperparameter-guide)
- [Reproducing Paper Results](#reproducing-paper-results)
- [API Usage (Importing as Modules)](#api-usage-importing-as-modules)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Installation

### Requirements

- Python 3.9+
- CUDA-enabled GPU (recommended; CPU works but is slow)
- ~16 GB GPU RAM for 2B–3B models, ~24 GB for 7B models

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/inference-time-alignment.git
cd inference-time-alignment

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
