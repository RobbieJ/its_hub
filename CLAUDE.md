# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Development installation with uv (recommended)
uv sync --extra dev

# Alternative: pip installation
pip install -e ".[dev]"

# Production installation
pip install its_hub
```

### DGX Spark Setup (ARM64 / CUDA 13.0)
```bash
# For NVIDIA DGX Spark (ARM64, Grace Blackwell, CUDA 13.0)

# 1. Validate environment
bash scripts/dgx_spark_validate.sh

# 2. Automated setup
bash scripts/dgx_spark_setup.sh

# 3. Manual verification
uname -m                    # Should show: aarch64
nvcc --version             # Should show: CUDA 13.0
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Contribution
When commit or raising PR, never mention it is by ClaudeCode.
never say 🤖 Generated with [Claude Code](https://claude.ai/code)" in the commit statment, don't mention claude!

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_algorithms.py

# Run tests with coverage
uv run pytest tests/ --cov=its_hub

# Run tests with verbose output
uv run pytest tests/ -v
```

### Code Quality
```bash
# Run linter checks
uv run ruff check its_hub/

# Fix auto-fixable linting issues
uv run ruff check its_hub/ --fix

# Format code with ruff
uv run ruff format its_hub/
```

### Git Workflow
```bash
# Create commits with sign-off
git commit -s -m "commit message"

# For any git commits, always use the sign-off flag (-s)
```

### Running Examples
```bash
# Test basic functionality
python scripts/test_math_example.py

# Benchmark algorithms (see script help for full options)
python scripts/benchmark.py --help
```

### IaaS Service (Inference-as-a-Service)
```bash
# Start IaaS service
uv run its-iaas --host 0.0.0.0 --port 8108

# Or using justfile (if available)
just iaas-start

# Check service health
curl -s http://localhost:8108/v1/models | jq .

# Configure the service (example: self-consistency algorithm)
curl -X POST http://localhost:8108/configure \
  -H "Content-Type: application/json" \
  -d '{"endpoint": "http://localhost:8100/v1", "api_key": "NO_API_KEY", "model": "your-model-name", "alg": "self-consistency"}'

# For comprehensive IaaS setup (multi-GPU, reward models, etc.), see docs/iaas-service.md
```

## Additional Tips
- Use `rg` in favor of `grep` whenever it's available
- Use `uv` for Python environment management: always start with `uv sync --extra dev` to init the env and run stuff with `uv run`
- In case of dependency issues during testing, try commenting out `reward_hub` and `vllm` temporarily in @pyproject.toml and retry.

## Architecture Overview

**its_hub** is a library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks. The core architecture uses abstract base classes to define clean interfaces between components.

### Key Base Classes (`its_hub/base.py`)
- `AbstractLanguageModel`: Interface for LM generation and evaluation
- `AbstractScalingAlgorithm`: Base for all scaling algorithms with unified `infer()` method
- `AbstractScalingResult`: Base for algorithm results with `the_one` property
- `AbstractOutcomeRewardModel`: Interface for outcome-based reward models
- `AbstractProcessRewardModel`: Interface for process-based reward models (step-by-step scoring)

### Main Components

#### Language Models (`its_hub/lms.py`)
- `OpenAICompatibleLanguageModel`: Primary LM implementation supporting vLLM and OpenAI APIs
- `StepGeneration`: Handles incremental generation with configurable step tokens and stop conditions
- Supports async generation with concurrency limits and backoff strategies

#### Algorithms (`its_hub/algorithms/`)
All algorithms follow the same interface: `infer(lm, prompt, budget, return_response_only=True)`

- **Self-Consistency**: Generate multiple responses, select most common answer
- **Best-of-N**: Generate N responses, select highest scoring via outcome reward model  
- **Beam Search**: Step-by-step generation with beam width, uses process reward models
- **Particle Filtering/Gibbs**: Probabilistic resampling with process reward models

#### Integration (`its_hub/integration/`)
- `LocalVllmProcessRewardModel`: Integrates with reward_hub library for process-based scoring
- `iaas.py`: Inference-as-a-Service FastAPI server providing OpenAI-compatible chat completions API with budget parameter for inference-time scaling

### Budget Interpretation
The budget parameter controls computational resources allocated to each algorithm. Different algorithms interpret budget as follows:
- **Self-Consistency/Best-of-N**: Number of parallel generations to create
- **Beam Search**: Total generations divided by beam width (controls search depth)
- **Particle Filtering**: Number of particles maintained during sampling

### Step Generation Pattern
The `StepGeneration` class enables incremental text generation:
- Configure step tokens (e.g., "\n\n" for reasoning steps)
- Set max steps and stop conditions
- Post-processing for clean output formatting

### Typical Workflow
1. Start vLLM server with instruction model
2. Initialize `OpenAICompatibleLanguageModel` pointing to server
3. Create `StepGeneration` with step/stop tokens appropriate for the task
4. Initialize reward model (e.g., `LocalVllmProcessRewardModel`)
5. Create scaling algorithm with step generation and reward model
6. Call `infer()` with prompt and budget

### Mathematical Focus
The library is optimized for mathematical reasoning:
- Predefined system prompts in `its_hub/utils.py` (SAL_STEP_BY_STEP_SYSTEM_PROMPT, QWEN_SYSTEM_PROMPT)
- Regex patterns for mathematical notation (e.g., `r"\boxed"` for final answers)
- Integration with math_verify for evaluation
- Benchmarking on MATH500 and AIME-2024 datasets

## Inference-as-a-Service (IaaS)

The its_hub library includes an IaaS service that provides OpenAI-compatible API with inference-time scaling capabilities. For comprehensive setup instructions, usage examples, and troubleshooting, see [docs/iaas-service.md](./docs/iaas-service.md).