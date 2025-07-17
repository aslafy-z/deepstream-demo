#!/usr/bin/env python3

"""
Docker Deployment Test Script
Tests container functionality and GPU access
"""

import subprocess
import time
import requests
import sys
import os
from pathlib import Path

def run_command(cmd, timeout=30):
    """Run shell command and return result"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def test_docker_availability():
    """Test if Docker is available and running"""
    print("Testing Docker availability...")
    
    success, stdout, stderr = run_command("docker --version")
    if not success:
        print(f"❌ Docker not available: {stderr}")
        return False
    
    print(f"✅ Docker version: {stdout.strip()}")
    
    success, stdout, stderr = run_command("docker info")
    if not success:
        print(f"❌ Docker daemon not running: {stderr}")
        return False
    
    print("✅ Docker daemon is running")
    return True

def test_nvidia_docker():
    """Test NVIDIA Docker runtime"""
    print("Testing NVIDIA Docker runtime...")
    
    success, stdout, stderr = run_command("docker info | grep nvidia")
    if not success:
        print("⚠️  NVIDIA Docker runtime not detected")
        return False
    
    print("✅ NVIDIA Docker runtime available")
    
    # Test GPU access
    success, stdout, stderr = run_command(
        "docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi", 
        timeout=60
    )
    if not success:
        print(f"⚠️  GPU access test failed: {stderr}")
        return False
    
    print("✅ GPU access confirmed")
    return True

def test_container_build():
    """Test container build process"""
    print("Testing container build...")
    
    # Check if Dockerfile exists
    if not Path("Dockerfile").exists():
        print("❌ Dockerfile not found")
        return False
    
    print("✅ Dockerfile found")
    
    # Build container (this might take a while)
    print("Building container (this may take several minutes)...")
    success, stdout, stderr = run_command(
        "docker build -t deepstream-detection-test:latest .", 
        timeout=1800  # 30 minutes
    )
    
    if not success:
        print(f"❌ Container build failed: {stderr}")
        return False
    
    print("✅ Container built successfully")
    return True

def test_container_run():
    """Test container execution"""
    print("Testing container execution...")
    
    # Run container with health check
    cmd = """
    docker run -d --name deepstream-test \
        --gpus all \
        -v $(pwd)/configs:/app/configs \
        deepstream-detection-test:latest \
        python3 -c "
import time
import sys
sys.path.append('/app/src')
try:
    from config_manager import ConfigManager
    print('Configuration manager imported successfully')
    time.sleep(10)
    print('Container test completed successfully')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"
    """
    
    success, stdout, stderr = run_command(cmd)
    if not success:
        print(f"❌ Container start failed: {stderr}")
        return False
    
    container_id = stdout.strip()
    print(f"✅ Container started: {container_id}")
    
    # Wait for container to complete
    time.sleep(15)
    
    # Check container logs
    success, stdout, stderr = run_command(f"docker logs {container_id}")
    if success and "Container test completed successfully" in stdout:
        print("✅ Container execution test passed")
        result = True
    else:
        print(f"❌ Container execution test failed: {stdout} {stderr}")
        result = False
    
    # Cleanup
    run_command(f"docker rm -f {container_id}")
    
    return result

def test_docker_compose():
    """Test Docker Compose functionality"""
    print("Testing Docker Compose...")
    
    if not Path("docker-compose.yml").exists():
        print("❌ docker-compose.yml not found")
        return False
    
    print("✅ docker-compose.yml found")
    
    # Test compose file syntax
    success, stdout, stderr = run_command("docker-compose config")
    if not success:
        print(f"❌ Docker Compose configuration invalid: {stderr}")
        return False
    
    print("✅ Docker Compose configuration valid")
    return True

def cleanup():
    """Clean up test resources"""
    print("Cleaning up test resources...")
    
    # Remove test containers and images
    run_command("docker rm -f deepstream-test 2>/dev/null")
    run_command("docker rmi deepstream-detection-test:latest 2>/dev/null")
    
    print("✅ Cleanup completed")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Docker Deployment Test Suite")
    print("=" * 60)
    
    tests = [
        ("Docker Availability", test_docker_availability),
        ("NVIDIA Docker Runtime", test_nvidia_docker),
        ("Container Build", test_container_build),
        ("Container Execution", test_container_run),
        ("Docker Compose", test_docker_compose),
    ]
    
    passed = 0
    total = len(tests)
    
    try:
        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
    
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    
    finally:
        cleanup()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Docker deployment is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())