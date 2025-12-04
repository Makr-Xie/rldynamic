# RL Dynamics - PPO Training with Component Reward Analysis

This repository contains custom implementations of PPO (Proximal Policy Optimization) training with gradient analysis and reward component tracking for mathematical reasoning tasks.

## 📁 Project Structure

```
rldynamics/
├── custom_training/          # Custom PPO trainer implementations
│   ├── custom_main_ppo.py    # Gradient similarity analysis version
│   └── custom_main_ppo_2.py  # Reward component tracking version
├── utils/                    # Utility functions
│   ├── rewards.py            # Reward computation functions
│   ├── download_model.py     # Model downloading utilities
│   └── process_math500.py    # Data processing for MATH500 dataset
├── scripts/                  # Training scripts
│   └── run_grpo.sh          # GRPO training launch script
├── verl/                    # veRL framework (submodule)
├── models/                  # Pre-trained models
├── checkpoints/             # Training checkpoints
├── gradient_analysis_math500/  # Gradient analysis outputs
└── reward_tracking_math500/    # Reward tracking outputs
```

## Setting up normally
If you have unrestricted access to your machine please follow the verl installation instructions here: https://verl.readthedocs.io/en/latest/start/install.html
With FSDP and VLLM options.


## Setting up workspace on delta cluster


#### 1. Pull the VERL image (using apptainer)
```
apptainer pull xxPathxx/verl-base_0.6.sif docker://verlai/verl:base-verl0.6-cu128-cudnn9.8-torch2.8.0-fa2.7.4
```
- move the .sif file wherever you want

#### 2. Set environment variables (example)
```
BASE=xxx/akunte2
PROJECT=$BASE/cs498repo/rldynamic
VERL_SRC=$BASE/cs498repo/verl
SIF=$BASE/verl-base_0.6.sif
```

#### 3. Create directories for persistent storage; (models, logs, pip cache, venv) ---
```
mkdir -p "$BASE/hf_cache" "$BASE/model_ckpts" "$BASE/runs_logs" "$BASE/pip_cache" "$BASE/.venvs"
```
#### 4. Clone repo : https://github.com/Makr-Xie/rldynamic.git
- Clone repo One level outside rldynamic
```
cs498repo/
|
| - verl/
| - rldynamic/

```
#### 5. Run Apptainer command once (AFTER REQUESTING COMPUTE FROM CLUSTER)

- Make sure you cd into your 'workspace' directory first

```
apptainer exec --cleanenv --nv \
  -B "$BASE":/mnt/user \
  --env HF_HOME=/mnt/user/hf_cache \
  --env XDG_CACHE_HOME=/mnt/user/hf_cache \
  --env TORCH_HOME=/mnt/user/hf_cache/torch \
  --env PIP_CACHE_DIR=/mnt/user/pip_cache \
  --pwd "$PROJECT" \
  "$SIF" \
  bash
```

#### 5.1 Run Bash commands:

```
python3 -m venv /mnt/user/.venvs/verl
source /mnt/user/.venvs/verl/bin/activate
cd verl
pip3 install -e .

python - <<PY
import pathlib, torch, verl
print("✅ VERL:", pathlib.Path(verl.__file__).resolve())
print("✅ GPU :", torch.cuda.get_device_name(0))
PY
```

### Done! Now when you run Apptainer again you can just do (example using my directories):

```
#assume that xxxx is the base directory
export IMG=xxxx/akunte2/verl-base_0.6.sif
export BASE=xxxx
export PROJECT=$BASE/cs498repo/rldynamic
apptainer exec --cleanenv --nv \
  -B "$BASE":/mnt/user \
  --env HF_HOME=/mnt/user/hf_cache \
  --env TRANSFORMERS_CACHE=/mnt/user/hf_cache \
  --env XDG_CACHE_HOME=/mnt/user/hf_cache \
  --env TORCH_HOME=/mnt/user/hf_cache/torch \
  --env PIP_CACHE_DIR=/mnt/user/pip_cache \
  --pwd "$PROJECT" \
  "$IMG" \
  bash -lc '. /mnt/user/.venvs/verl/bin/activate; exec bash'
```

### Then you should be in the Apptainer environment, and you can confirm that verl,torch,gpu are working by doing (after running 'python' inside the apptainer container):

```
import torch
import verl
print(torch.cuda.get_device_name(0), verl.__file__)
```



## 🎯 Custom Training Implementations

### 1. `custom_main_ppo.py` - Gradient Similarity Analysis

**Purpose**: Analyzes gradient similarity between different reward components (correctness, format, length) during PPO training.

**Key Features**:
- Inherits from `RayPPOTrainer` via monkey patching
- Computes component-specific advantages for each reward type
- Calculates gradients for each component separately
- Computes cosine similarity between component gradients
- Saves gradient data asynchronously to `.npz` files

**Architecture**:
```python
CustomRayPPOTrainer (extends RayPPOTrainer)
├── __init__()
│   └── Initialize gradient tracking variables
├── _wrap_update_actor()
│   ├── Extract batch data (prompts, responses, ground truth)
│   ├── Compute component rewards (correctness, format, length)
│   ├── Convert to token-level rewards
│   ├── Compute component-specific advantages
│   ├── Compute component-specific gradients
│   ├── Calculate gradient similarity (cosine)
│   └── Save gradients asynchronously
├── wait_for_saves()
│   └── Wait for background save threads to complete
└── fit()
    └── Wrap update_actor and run training
```

**Reward Conversion**:
- **Correctness**: Last token only (binary reward)
- **Format**: Last token only (or distributed across "####" + answer tokens)
- **Length**: Averaged across all tokens

**Output**: `gradient_analysis_math500/gradients_step_X.npz`

---

### 2. `custom_main_ppo_2.py` - Reward Component Tracking

**Purpose**: Tracks correctness, format, and length rewards for each training step without gradient computation.

**Key Features**:
- Lightweight version without gradient analysis
- Records reward components for every rollout
- Tracks both reward scores and actual lengths
- Saves all data to a single JSON file
- Real-time statistics printing

**Architecture**:
```python
CustomRayPPOTracker (extends RayPPOTrainer)
├── __init__()
│   ├── Initialize reward_history dict
│   └── Create save directory
├── _wrap_update_actor()
│   ├── Extract outputs and ground truth
│   ├── Compute component rewards + actual lengths
│   ├── Build step_data with statistics
│   ├── Append to reward_history
│   ├── Save to JSON
│   └── Print statistics
├── _save_rewards()
│   └── Save reward_history to JSON with timestamp
└── fit()
    └── Wrap update_actor and run training
```

**Data Recorded**:
```json
{
  "metadata": {
    "created_at": "2025-11-05 05:47:00",
    "model": "qwen3_4b",
    "dataset": "gsm8k",
    "last_saved": "2025-11-05 06:15:23"
  },
  "steps": [
    {
      "step": 0,
      "timestamp": "2025-11-05 05:47:10",
      "num_samples": 64,
      "statistics": {
        "avg_correctness": 0.75,
        "avg_format": 0.65,
        "avg_length_reward": 0.80,
        "avg_actual_length": 724.5,
        "correct_count": 48,
        "format_valid_count": 52,
        "total_samples": 64
      },
      "samples": [
        {
          "idx": 0,
          "correctness": 1.0,
          "format": 0.8,
          "length_reward": 0.75,
          "actual_length": 856
        },
        ...
      ]
    }
  ]
}
```

**Output**: `reward_tracking_math500/all_rewards.json`

---

## 🛠️ Utils Folder

### `rewards.py` - Reward Computation

**Core Functions**:

#### 1. `compute_correctness_score(solution_str, ground_truth)`
- Extracts answer from `\boxed{}` notation
- Compares with ground truth
- Returns: `1.0` (correct) or `0.0` (incorrect)

#### 2. `compute_format_score(solution_str)`
- Checks for step-by-step reasoning indicators:
  - "first", "second", "third" keywords (+0.25 each)
  - "step1", "step2" patterns (+0.25 each)
  - `\boxed` notation (+0.4)
- Returns: Score in `[0.0, 1.0]`

#### 3. `compute_length_score(solution_str, max_length=800)`
- Penalizes responses that are too short or too long
- Formula: `1 - |len(solution) / max_length - 1|`
- Returns: Score in `[0.0, 1.0]`

#### 4. `compute_component_scores(data_source, solution_str, ground_truth, extra_info, weights)`
- Computes all three components
- Returns dictionary with:
  - `correctness_binary`, `format_binary`, `length_binary`
  - `correctness_weighted`, `format_weighted`, `length_weighted`
  - `weights`, `total`

**Default Weights**:
```python
DEFAULT_WEIGHTS = {
    'correctness': 0.34,
    'format': 0.33,
    'length': 0.33
}
```

**Helper Functions**:
- `extract_solution()`: Extracts answer from `\boxed{}`
- `last_boxed_only_string()`: Finds last `\boxed{}` in text
- `remove_boxed()`: Removes `\boxed{}` wrapper
- `strip_string()`: Normalizes mathematical expressions

---

### Other Utils

- **`download_model.py`**: Utilities for downloading pre-trained models from Hugging Face
- **`process_math500.py`**: Data processing scripts for MATH500 dataset

---

## 📜 Scripts Folder

### `run_grpo.sh` - GRPO Training Script

**Purpose**: Launch GRPO (Group Relative Policy Optimization) training with custom configurations.

**Key Configurations**:

```bash
# Algorithm
algorithm.adv_estimator=grpo

# Data
data.train_files=/path/to/math500/train.parquet
data.val_files=/path/to/math500/test.parquet
data.train_batch_size=32
data.max_prompt_length=512
data.max_response_length=1024

# Model
actor_rollout_ref.model.path=/path/to/Qwen2.5-3B-Instruct
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.model.enable_gradient_checkpointing=True

# PPO Settings
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001

# Rollout (vLLM)
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.rollout.n=8  # Number of samples per prompt

# Training
trainer.n_gpus_per_node=8
trainer.total_epochs=10
trainer.save_freq=20
trainer.test_freq=5

# Custom Reward Function
custom_reward_function.path=/path/to/utils/rewards.py
custom_reward_function.name=compute_correctness_score
```

**Usage**:
```bash
bash scripts/run_grpo.sh
```

---

## 🚀 How to Use

### 1. Training with Gradient Analysis
```bash
# Modify the main script to import custom_main_ppo
python -m custom_training.custom_main_ppo \
    --config config/ppo_trainer.yaml

# Output: gradient_analysis_math500/gradients_step_X.npz
```

### 2. Training with Reward Tracking
```bash
# Modify the main script to import custom_main_ppo_2
python -m custom_training.custom_main_ppo_2 \
    --config config/ppo_trainer.yaml

# Output: reward_tracking_math500/all_rewards.json
```

### 3. Using run_grpo.sh
```bash
# Launch GRPO training
bash scripts/run_grpo.sh

# The script will use the custom trainer specified in the imports
```

---

## 📊 Output Analysis

### Gradient Analysis Output
```python
import numpy as np

# Load gradient data
data = np.load('gradient_analysis_math500/gradients_step_0.npz')

# Access component gradients
correctness_grads = {k: v for k, v in data.items() if k.startswith('correctness/')}
format_grads = {k: v for k, v in data.items() if k.startswith('format/')}
length_grads = {k: v for k, v in data.items() if k.startswith('length/')}
```

### Reward Tracking Output
```python
import json

# Load reward data
with open('reward_tracking_math500/all_rewards.json', 'r') as f:
    data = json.load(f)

# Analyze trends
steps = data['steps']
correctness_trend = [s['statistics']['avg_correctness'] for s in steps]
format_trend = [s['statistics']['avg_format'] for s in steps]
length_trend = [s['statistics']['avg_actual_length'] for s in steps]
```

---

## 🔧 Key Design Patterns

### Monkey Patching
Both custom trainers use monkey patching to inject custom functionality:

```python
import verl.trainer.ppo.ray_trainer
from verl.trainer import main_ppo

# Replace the original trainer
verl.trainer.ppo.ray_trainer.RayPPOTrainer = CustomRayPPOTrainer
main_ppo.RayPPOTrainer = CustomRayPPOTrainer
```

### Reward Component Decomposition
- **Correctness**: Task-specific (answer matching)
- **Format**: Process reward (step-by-step reasoning)
- **Length**: Length control (avoid too short/long responses)

### Asynchronous Saving (custom_main_ppo.py)
```python
import threading

def save_gradients_async(data, file_path):
    np.savez_compressed(file_path, **data)

thread = threading.Thread(target=save_gradients_async, args=(save_data, filename))
thread.start()
```
