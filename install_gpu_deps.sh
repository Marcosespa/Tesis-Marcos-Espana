#!/bin/bash

# GPU Dependencies Installation Script
# This script installs the required dependencies for GPU-accelerated processing

set -e

echo "🚀 Installing GPU-optimized dependencies for DatosTesis project..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
    echo "   CUDA Version: $CUDA_VERSION"
    
    # Install CUDA-specific requirements
    echo "📦 Installing CUDA-optimized packages..."
    pip install -r requirements-gpu.txt
    
    # Verify PyTorch CUDA installation
    echo "🔍 Verifying PyTorch CUDA installation..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
    
else
    echo "⚠️  CUDA not detected. Installing CPU-only version..."
    pip install -r requirements-pip.txt
    
    # Verify PyTorch installation
    echo "🔍 Verifying PyTorch installation..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
fi

# Install additional system dependencies if needed
echo "📦 Installing system dependencies..."

# Check if we're on Ubuntu/Debian
if command -v apt-get &> /dev/null; then
    echo "Installing system packages for Ubuntu/Debian..."
    sudo apt-get update
    sudo apt-get install -y \
        tesseract-ocr \
        tesseract-ocr-eng \
        libtesseract-dev \
        build-essential \
        python3-dev
fi

# Check if we're on CentOS/RHEL
if command -v yum &> /dev/null; then
    echo "Installing system packages for CentOS/RHEL..."
    sudo yum install -y \
        tesseract \
        tesseract-langpack-eng \
        gcc \
        python3-devel
fi

echo "✅ Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Test GPU availability: python -c 'import torch; print(torch.cuda.is_available())'"
echo "2. Test CrewAI installation: python -c 'import crewai; print(\"CrewAI installed successfully\")'"
echo "3. Test Weaviate connection: python weaviate/conection_test.py"
echo "4. Run the pipeline: python pipeline.py"
echo ""
echo "🔧 For troubleshooting:"
echo "- Check CUDA installation: nvidia-smi"
echo "- Check PyTorch CUDA: python -c 'import torch; print(torch.cuda.is_available())'"
echo "- Check GPU memory: nvidia-smi --query-gpu=memory.total,memory.used --format=csv"

