#!/bin/bash
# DGX Spark Setup Script for its_hub
# This script sets up its_hub on DGX Spark with CUDA 13 compatibility
# Usage: bash dgx_spark_setup.sh

set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
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

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Configuration
WORK_DIR="${HOME}/its_hub_testing"
REPO_URL="https://github.com/RobbieJ/its_hub.git"
BRANCH="claude/cuda-13-compatibility-011CUuPMdsZQWQptMNGGSoQo"

print_section "DGX Spark Setup for its_hub"

echo "This script will:"
echo "  1. Create working directory: $WORK_DIR"
echo "  2. Clone its_hub repository"
echo "  3. Checkout compatibility branch"
echo "  4. Install uv (if needed)"
echo "  5. Set up Python environment"
echo "  6. Install dependencies"
echo "  7. Run tests"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 1
fi

# 1. Create working directory
print_section "1. Creating Working Directory"

if [ -d "$WORK_DIR" ]; then
    print_warning "Directory $WORK_DIR already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$WORK_DIR"
        mkdir -p "$WORK_DIR"
        print_success "Directory recreated"
    else
        print_info "Using existing directory"
    fi
else
    mkdir -p "$WORK_DIR"
    print_success "Directory created: $WORK_DIR"
fi

cd "$WORK_DIR"

# 2. Clone repository
print_section "2. Cloning its_hub Repository"

if [ -d "its_hub/.git" ]; then
    print_warning "Repository already cloned"
    cd its_hub
    git fetch origin
    print_success "Repository updated"
else
    git clone "$REPO_URL"
    cd its_hub
    print_success "Repository cloned"
fi

# 3. Checkout branch
print_section "3. Checking Out Compatibility Branch"

git checkout "$BRANCH"
git pull origin "$BRANCH" || print_warning "Could not pull latest changes"
print_success "Branch checked out: $BRANCH"

# 4. Install uv if needed
print_section "4. Installing uv Package Manager"

if command -v uv &> /dev/null; then
    print_success "uv already installed: $(uv --version)"
else
    print_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    if command -v uv &> /dev/null; then
        print_success "uv installed: $(uv --version)"
    else
        print_error "uv installation failed"
        print_info "Falling back to pip installation method"
    fi
fi

# 5. Set up Python environment
print_section "5. Setting Up Python Environment"

print_info "Python version: $(python3 --version)"
print_info "CUDA version: $(nvcc --version | grep release || echo 'nvcc not found')"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    print_success "ARM64 architecture detected (DGX Spark)"
    export ARCH_TYPE="arm64"
else
    print_warning "x86_64 architecture detected (unexpected for DGX Spark)"
    export ARCH_TYPE="x86_64"
fi

# 6. Install dependencies
print_section "6. Installing Dependencies"

if command -v uv &> /dev/null; then
    print_info "Installing with uv (recommended)..."

    # Try dev installation first (includes all extras)
    if uv sync --extra dev; then
        print_success "Development dependencies installed"
    else
        print_warning "Dev installation failed, trying core only..."
        if uv sync; then
            print_success "Core dependencies installed"
        else
            print_error "Installation failed with uv"
            print_info "Check error messages above"
            exit 1
        fi
    fi
else
    print_info "Installing with pip..."

    # Try dev installation
    if pip3 install -e ".[dev]"; then
        print_success "Development dependencies installed"
    else
        print_warning "Dev installation failed, trying core only..."
        if pip3 install -e .; then
            print_success "Core dependencies installed"
        else
            print_error "Installation failed with pip"
            exit 1
        fi
    fi
fi

# 7. Verify installation
print_section "7. Verifying Installation"

echo "Checking its_hub..."
if python3 -c "import its_hub" 2>/dev/null; then
    print_success "its_hub imported successfully"
else
    print_error "Cannot import its_hub"
    exit 1
fi

echo ""
echo "Checking PyTorch..."
python3 << 'PYEOF'
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
except ImportError:
    print("⚠️  PyTorch not available")
PYEOF

echo ""
echo "Checking vLLM..."
python3 << 'PYEOF'
try:
    import vllm
    print(f"✅ vLLM available")
except ImportError:
    print("⚠️  vLLM not available (required for PRM algorithms)")
PYEOF

echo ""
echo "Checking reward_hub..."
python3 << 'PYEOF'
try:
    import reward_hub
    print(f"✅ reward_hub available")
except ImportError:
    print("⚠️  reward_hub not available")
PYEOF

# 8. Run tests
print_section "8. Running Tests"

read -p "Run test suite? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v uv &> /dev/null; then
        print_info "Running tests with uv..."
        uv run pytest tests/ -v || print_warning "Some tests failed"
    else
        print_info "Running tests with pytest..."
        python3 -m pytest tests/ -v || print_warning "Some tests failed"
    fi
else
    print_info "Skipping tests"
fi

# Summary
print_section "Setup Complete!"

echo ""
echo "Working directory: $WORK_DIR/its_hub"
echo "Branch: $BRANCH"
echo ""
echo "Next steps:"
echo "  cd $WORK_DIR/its_hub"
echo "  uv run pytest tests/              # Run tests"
echo "  uv run python scripts/test_math_example.py  # Test basic functionality"
echo ""
print_success "DGX Spark setup completed successfully!"
