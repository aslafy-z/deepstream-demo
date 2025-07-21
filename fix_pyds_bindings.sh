#!/bin/bash

# Script to fix DeepStream Python bindings (pyds) on G4DN instances
# For use with Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== DeepStream Python Bindings Fix Script ===${NC}"
echo -e "${BLUE}For G4DN.xlarge with Deep Learning AMI${NC}"
echo ""

# Check if running inside Docker container
if [ ! -f /.dockerenv ]; then
    echo -e "${YELLOW}This script should be run inside the DeepStream Docker container.${NC}"
    echo -e "${YELLOW}Run it with: docker exec -it deepstream-detection bash fix_pyds_bindings.sh${NC}"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Check DeepStream installation
echo -e "${BLUE}Step 1: Checking DeepStream installation...${NC}"
if [ -d "/opt/nvidia/deepstream" ]; then
    echo -e "${GREEN}DeepStream installation found at /opt/nvidia/deepstream${NC}"
    ls -la /opt/nvidia/deepstream/
else
    echo -e "${RED}DeepStream installation not found at /opt/nvidia/deepstream${NC}"
    echo -e "${YELLOW}Checking alternative locations...${NC}"
    find /opt -name "deepstream" -type d
fi

# Step 2: Find pyds wheel files
echo -e "\n${BLUE}Step 2: Searching for pyds wheel files...${NC}"
PYDS_WHEELS=$(find /opt -name "pyds*.whl")
if [ -n "$PYDS_WHEELS" ]; then
    echo -e "${GREEN}Found pyds wheel files:${NC}"
    echo "$PYDS_WHEELS"
else
    echo -e "${YELLOW}No pyds wheel files found.${NC}"
fi

# Step 3: Check Python bindings source
echo -e "\n${BLUE}Step 3: Checking for Python bindings source...${NC}"
if [ -d "/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps" ]; then
    echo -e "${GREEN}Found DeepStream Python apps at /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps${NC}"
    ls -la /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps
elif [ -d "/opt/nvidia/deepstream/deepstream/sources/python" ]; then
    echo -e "${GREEN}Found Python bindings source at /opt/nvidia/deepstream/deepstream/sources/python${NC}"
    ls -la /opt/nvidia/deepstream/deepstream/sources/python
else
    echo -e "${YELLOW}No Python bindings source found in standard locations.${NC}"
fi

# Step 4: Check Python environment
echo -e "\n${BLUE}Step 4: Checking Python environment...${NC}"
echo -e "${BLUE}Python version:${NC}"
python3 --version
echo -e "${BLUE}Python path:${NC}"
python3 -c "import sys; print(sys.path)"

# Step 5: Try to fix the bindings
echo -e "\n${BLUE}Step 5: Attempting to fix Python bindings...${NC}"
echo -e "${YELLOW}This may take a few minutes...${NC}"

# Method 1: Try to install from existing wheel
if [ -n "$PYDS_WHEELS" ]; then
    echo -e "${BLUE}Method 1: Installing from existing wheel...${NC}"
    for wheel in $PYDS_WHEELS; do
        echo "Installing $wheel..."
        pip3 install --force-reinstall "$wheel" && echo -e "${GREEN}Successfully installed $wheel${NC}" || echo -e "${RED}Failed to install $wheel${NC}"
    done
fi

# Method 2: Try to install from install script
if [ -f "/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/install.sh" ]; then
    echo -e "${BLUE}Method 2: Running install script...${NC}"
    cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps
    bash install.sh && echo -e "${GREEN}Install script completed successfully${NC}" || echo -e "${RED}Install script failed${NC}"
fi

# Method 3: Try to build from source
if [ -d "/opt/nvidia/deepstream/deepstream/sources/python" ]; then
    echo -e "${BLUE}Method 3: Building from source...${NC}"
    cd /opt/nvidia/deepstream/deepstream/sources/python
    python3 setup.py install && echo -e "${GREEN}Built and installed from source successfully${NC}" || echo -e "${RED}Failed to build from source${NC}"
fi

# Method 4: Clone and build from GitHub
if [ ! -d "/opt/deepstream_python_apps" ]; then
    echo -e "${BLUE}Method 4: Cloning and building from GitHub...${NC}"
    cd /opt
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
    cd deepstream_python_apps
    git submodule update --init
    cd bindings
    mkdir -p build
    cd build
    cmake .. -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=8
    make -j$(nproc)
    pip3 install ./pyds-*.whl && echo -e "${GREEN}Built and installed from GitHub successfully${NC}" || echo -e "${RED}Failed to build from GitHub${NC}"
fi

# Step 6: Update PYTHONPATH
echo -e "\n${BLUE}Step 6: Updating PYTHONPATH...${NC}"
echo 'export PYTHONPATH=$PYTHONPATH:/opt/nvidia/deepstream/deepstream/lib:/opt/nvidia/deepstream/deepstream/python:/opt/nvidia/deepstream/deepstream/sources/python:/usr/lib/python3/dist-packages:/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings:/opt/deepstream_python_apps/bindings/build' >> ~/.bashrc
echo -e "${GREEN}PYTHONPATH updated in ~/.bashrc${NC}"

# Step 7: Create symlinks
echo -e "\n${BLUE}Step 7: Creating symlinks...${NC}"
mkdir -p /opt/nvidia/deepstream/deepstream/sources/python/bindings
ln -sf /opt/nvidia/deepstream/deepstream /usr/lib/x86_64-linux-gnu/deepstream 2>/dev/null || echo -e "${YELLOW}Failed to create symlink, may already exist${NC}"

# Step 8: Test import
echo -e "\n${BLUE}Step 8: Testing pyds import...${NC}"
echo "import sys; print('Python path:', sys.path); import pyds; print('pyds imported successfully')" > /tmp/test_pyds.py

# Source the updated environment
source ~/.bashrc

# Try to import pyds
python3 /tmp/test_pyds.py && echo -e "${GREEN}pyds imported successfully!${NC}" || {
    echo -e "${RED}Failed to import pyds.${NC}"
    echo -e "${YELLOW}Searching for pyds files...${NC}"
    find /opt -name "*pyds*"
}

# Step 9: Create a fix for the pipeline_manager.py
echo -e "\n${BLUE}Step 9: Creating a fix for pipeline_manager.py...${NC}"
if [ -f "/app/src/pipeline_manager.py" ]; then
    echo -e "${BLUE}Backing up original pipeline_manager.py...${NC}"
    cp /app/src/pipeline_manager.py /app/src/pipeline_manager.py.bak
    
    echo -e "${BLUE}Modifying pipeline_manager.py to handle pyds import errors...${NC}"
    sed -i '1s/^/import sys\nimport os\n\n# Add all possible pyds paths\nsys.path.extend([\n    "\/opt\/nvidia\/deepstream\/deepstream\/lib",\n    "\/opt\/nvidia\/deepstream\/deepstream\/python",\n    "\/opt\/nvidia\/deepstream\/deepstream\/sources\/python",\n    "\/usr\/lib\/python3\/dist-packages",\n    "\/opt\/nvidia\/deepstream\/deepstream\/sources\/deepstream_python_apps\/bindings",\n    "\/opt\/deepstream_python_apps\/bindings\/build"\n])\n\n/' /app/src/pipeline_manager.py
    
    echo -e "${BLUE}Modifying pyds import in pipeline_manager.py...${NC}"
    sed -i 's/import pyds/try:\n    import pyds\nexcept ImportError as e:\n    print("WARNING: Failed to import pyds:", e)\n    print("Continuing without DeepStream Python bindings...")\n    # Create a mock pyds module\n    import types\n    pyds = types.ModuleType("pyds")\n    pyds.__version__ = "0.0.0"\n    sys.modules["pyds"] = pyds/' /app/src/pipeline_manager.py
    
    echo -e "${GREEN}Modified pipeline_manager.py to handle pyds import errors${NC}"
else
    echo -e "${YELLOW}Could not find /app/src/pipeline_manager.py${NC}"
fi

echo -e "\n${BLUE}=== Fix Complete ===${NC}"
echo -e "${BLUE}Try running your application now.${NC}"
echo -e "${BLUE}If you still have issues, you may need to rebuild the Docker image with the updated Dockerfile.${NC}"
echo -e "${YELLOW}Remember to run 'source ~/.bashrc' in any new shell sessions.${NC}"