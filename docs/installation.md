# Installation

## Prerequisites

- Python 3.10+ (3.11+ recommended)
- pip or uv package manager
- GPU with CUDA 11.8+ (CUDA 12.8+ or 13.0 recommended for DGX Spark and latest GPUs)
  - For PRM installation only
  - CUDA 13.0 required for NVIDIA DGX Spark (ARM64/Grace Blackwell)

## Installation Options

| Option | Command | Use Case |
|--------|---------|----------|
| **Core** | `pip install its_hub` | Best-of-N, Self-Consistency, cloud APIs |
| **PRM** | `pip install its_hub[prm]` | Particle Filtering, Beam Search, local reward models |
| **Cloud** | `pip install its_hub[cloud]` | AWS Bedrock, Google Vertex AI |
| **Research** | `pip install its_hub[research]` | Benchmarks, evaluation tools |
| **Dev** | `pip install -e ".[dev]"` | Contributing, testing |

---

## Core Installation

```bash
pip install its_hub
```

### What's Included

**Algorithms**: Best-of-N, Self-Consistency, LLM Judge
**Language Models**: OpenAI-compatible, LiteLLM (100+ providers)
**Key Dependencies**: `openai`, `litellm`, `reward-hub`, `transformers`, `fastapi`

### When to Use

**Use if**: Working with cloud APIs (OpenAI, Anthropic, etc.), no GPU needed
**Skip if**: Need Particle Filtering/Beam Search or local process reward models

### Under the Hood

- **Size**: ~50MB (no vLLM or CUDA dependencies)
- **Installation time**: 1-2 minutes
- **GPU required**: No
- **What's excluded**: vLLM, local reward model inference

```python
# Verify installation
from its_hub.algorithms import BestOfN, SelfConsistency
from its_hub.integration.reward_hub import LLMJudgeRewardModel
```

---

## Process Reward Model (PRM) Installation

```bash
pip install its_hub[prm]
```

### What's Added

**Algorithms**: Particle Filtering, Beam Search (+ all core algorithms)
**Reward Models**: `LocalVllmProcessRewardModel` for step-by-step scoring
**Additional Dependencies**: `reward-hub[prm]` (includes vLLM with pinned versions)

### When to Use

**Use if**: Need step-by-step reasoning with local reward models, have GPU
**Skip if**: Only using cloud APIs or outcome-based scoring

### Under the Hood

- **Size**: ~2-3GB (includes vLLM + CUDA dependencies)
- **Installation time**: 5-10 minutes
- **GPU required**: Yes (10-20GB VRAM for typical 7B reward models)
- **Version pinning**: `reward-hub[prm]` pins compatible vLLM + transformers + PyTorch versions
- **CUDA compatibility**:
  - CUDA 11.8+: Standard support
  - CUDA 12.8+: Latest vLLM releases
  - CUDA 13.0: DGX Spark and Blackwell GPUs (requires NVIDIA Container or compatible builds)

```python
# Verify installation
from its_hub.algorithms import ParticleFiltering, BeamSearch
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

# Check GPU
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
```

---

## Cloud Installation

```bash
pip install its_hub[cloud]
```

**Adds**: AWS Bedrock (`boto3`) and Google Vertex AI (`google-cloud-aiplatform`) SDKs
**Use if**: Need direct SDK access to Bedrock or Vertex AI (most cloud providers work with core via LiteLLM)

---

## Research Installation

```bash
pip install its_hub[research]
```

**Adds**: `math-verify`, `datasets`, `matplotlib`
**Use if**: Running benchmarks on MATH500/AIME or evaluating algorithm performance
**Includes**: Benchmark scripts in `scripts/benchmark.py`

---

## Development Installation

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub

# Recommended: uv
uv sync --extra dev

# Alternative: pip
pip install -e ".[dev]"
```

**Includes**: All core + PRM + `pytest`, `ruff`, `jupyter`, notebooks
**Use if**: Contributing, testing, or developing new features

```bash
# Run tests
uv run pytest tests/
uv run pytest tests/ --cov=its_hub

# Code quality
uv run ruff check its_hub/ --fix
uv run ruff format its_hub/
```

---

## Combining Extras

```bash
pip install its_hub[prm,research]           # PRM + benchmarking
pip install its_hub[cloud,research]          # Cloud + benchmarking
pip install -e ".[dev,research,cloud]"       # Everything
```

---

## Verification

```bash
# Core
python -c "from its_hub.algorithms import BestOfN; print('✅ Core OK')"

# PRM
python -c "from its_hub.algorithms import ParticleFiltering; print('✅ PRM OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## DGX Spark Installation

**NVIDIA DGX Spark** is a desktop AI supercomputer featuring ARM64 architecture (Grace Blackwell GB10) with CUDA 13.0 support. Special considerations apply:

### Quick Setup

```bash
# 1. Download and run validation script
curl -O https://raw.githubusercontent.com/Red-Hat-AI-Innovation-Team/its_hub/main/scripts/dgx_spark_validate.sh
bash dgx_spark_validate.sh

# 2. Download and run setup script
curl -O https://raw.githubusercontent.com/Red-Hat-AI-Innovation-Team/its_hub/main/scripts/dgx_spark_setup.sh
bash dgx_spark_setup.sh
```

### Manual Setup

```bash
# Clone repository
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub

# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install its_hub
uv sync --extra dev
```

### DGX Spark Requirements

- **Architecture**: ARM64 (aarch64)
- **CUDA**: 13.0 (pre-installed with DGX OS)
- **OS**: Ubuntu 22.04-based DGX OS
- **Memory**: 128GB unified memory
- **GPU**: GB10 Blackwell (6144 CUDA cores)

### Verification

```bash
# Check system
uname -m                    # Should show: aarch64
nvcc --version             # Should show: CUDA 13.0

# Test PyTorch
python3 << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
EOF

# Test its_hub
python3 -c "from its_hub.algorithms import BestOfN; print('✅ its_hub OK')"
```

### Troubleshooting

**Binary compatibility issues**: Ensure all packages are ARM64-compiled
```bash
# Check package architecture
python3 -c "import platform; print(platform.machine())"
```

**vLLM installation fails**: Use NVIDIA Container or pip with ARM64 wheels
```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130
```

**CUDA version mismatch**: Verify CUDA 13.0 toolkit is installed
```bash
ls /usr/local/cuda*/version.txt
cat /usr/local/cuda/version.txt
```

---

## Next Steps

- [Quick Start Guide](quick-start.md) - Best-of-N and Particle Filtering examples
- [IaaS Service Guide](iaas-service.md) - Deploy as OpenAI-compatible API
- [Development Guide](development.md) - Contributing guidelines

For runtime issues (CUDA OOM, server errors, etc.), see the troubleshooting sections in the Quick Start or IaaS Service guides.