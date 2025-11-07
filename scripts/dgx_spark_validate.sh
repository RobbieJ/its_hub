#!/bin/bash
# DGX Spark Validation Script for its_hub CUDA 13 Compatibility
# Usage: bash dgx_spark_validate.sh

set -e

echo "========================================"
echo "DGX Spark Validation for its_hub"
echo "========================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_section() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 1. System Information
print_section "1. System Information"

echo "Hostname:"
hostname

echo ""
echo "Kernel:"
uname -a

echo ""
echo "OS Release:"
cat /etc/os-release | grep -E "^(NAME|VERSION)="

echo ""
echo "CPU Info:"
lscpu | grep -E "Architecture|CPU\(s\)|Model name|Thread|Core"

# Check if ARM64
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    print_success "Architecture: ARM64 (Expected for DGX Spark)"
else
    print_warning "Architecture: $ARCH (Expected ARM64 for DGX Spark)"
fi

# 2. GPU and CUDA Information
print_section "2. GPU and CUDA Information"

if command -v nvidia-smi &> /dev/null; then
    print_success "nvidia-smi found"
    nvidia-smi
else
    print_error "nvidia-smi not found"
fi

echo ""
echo "CUDA Compiler Version:"
if command -v nvcc &> /dev/null; then
    nvcc --version
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    print_success "CUDA Version: $CUDA_VERSION"

    # Check if CUDA 13.0
    if [[ "$CUDA_VERSION" == 13.* ]]; then
        print_success "CUDA 13.x detected (Compatible with DGX Spark)"
    elif [[ "$CUDA_VERSION" == 12.* ]]; then
        print_warning "CUDA 12.x detected (May need upgrade to 13.0 for optimal DGX Spark support)"
    else
        print_warning "CUDA $CUDA_VERSION detected (Expected 13.0 for DGX Spark)"
    fi
else
    print_error "nvcc not found - CUDA toolkit not installed"
fi

# 3. Python Environment
print_section "3. Python Environment"

echo "Python Version:"
python3 --version
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python: $PYTHON_VERSION"

echo ""
echo "pip Version:"
pip3 --version

echo ""
echo "Python Path:"
which python3

# Check for uv
echo ""
if command -v uv &> /dev/null; then
    print_success "uv package manager found"
    uv --version
else
    print_warning "uv package manager not found (recommended for its_hub)"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# 4. Memory and Disk
print_section "4. Memory and Disk"

echo "Memory:"
free -h

echo ""
echo "Disk Space (home directory):"
df -h ~

# 5. Git
print_section "5. Git"

if command -v git &> /dev/null; then
    print_success "git found"
    git --version
else
    print_error "git not found"
fi

# 6. Check for existing PyTorch
print_section "6. Existing Python Packages"

echo "Checking for PyTorch..."
if python3 -c "import torch" 2>/dev/null; then
    python3 << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
PYEOF
    print_success "PyTorch installed and CUDA available"
else
    print_warning "PyTorch not installed or CUDA not available"
fi

echo ""
echo "Checking for vLLM..."
if python3 -c "import vllm" 2>/dev/null; then
    python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
    print_success "vLLM installed"
else
    print_warning "vLLM not installed"
fi

echo ""
echo "Checking for transformers..."
if python3 -c "import transformers" 2>/dev/null; then
    python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
    print_success "Transformers installed"
else
    print_warning "Transformers not installed"
fi

echo ""
echo "Checking for reward_hub..."
if python3 -c "import reward_hub" 2>/dev/null; then
    python3 -c "import reward_hub; print(f'reward_hub version: {reward_hub.__version__ if hasattr(reward_hub, \"__version__\") else \"unknown\"}')"
    print_success "reward_hub installed"
else
    print_warning "reward_hub not installed"
fi

# 7. Network
print_section "7. Network Connectivity"

echo "Testing internet connectivity..."
if curl -Is https://www.google.com | head -1 > /dev/null 2>&1; then
    print_success "Internet connection available"
else
    print_warning "Internet connection check failed"
fi

echo ""
echo "Testing GitHub connectivity..."
if curl -Is https://github.com | head -1 > /dev/null 2>&1; then
    print_success "GitHub accessible"
else
    print_warning "GitHub not accessible"
fi

echo ""
echo "Testing PyPI connectivity..."
if curl -Is https://pypi.org | head -1 > /dev/null 2>&1; then
    print_success "PyPI accessible"
else
    print_warning "PyPI not accessible"
fi

# 8. Summary
print_section "8. Validation Summary"

echo "System: $(uname -m)"
echo "OS: $(cat /etc/os-release | grep "^PRETTY_NAME=" | cut -d'"' -f2)"
echo "Python: $PYTHON_VERSION"
if command -v nvcc &> /dev/null; then
    echo "CUDA: $CUDA_VERSION"
else
    echo "CUDA: Not found"
fi

echo ""
print_success "Validation complete!"
echo ""
echo "Next steps:"
echo "1. Review the output above for any errors or warnings"
echo "2. Share this output for compatibility analysis"
echo "3. Proceed with its_hub installation and testing"
echo ""
