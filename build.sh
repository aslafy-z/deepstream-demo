#!/bin/bash

# Build script for RTSP Object Detection System Docker containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    print_success "Docker is available"
}

# Check if NVIDIA Docker runtime is available
check_nvidia_docker() {
    if docker info 2>/dev/null | grep -q nvidia; then
        print_success "NVIDIA Docker runtime is available"
        return 0
    else
        print_warning "NVIDIA Docker runtime not detected. GPU acceleration may not work."
        return 1
    fi
}

# Detect platform
detect_platform() {
    if [ -f /etc/nv_tegra_release ]; then
        echo "jetson"
    else
        echo "desktop"
    fi
}

# Build function
build_container() {
    local platform=$1
    local tag_suffix=""
    local dockerfile="Dockerfile"
    
    if [ "$platform" = "jetson" ]; then
        tag_suffix="-jetson"
        dockerfile="Dockerfile.jetson"
    fi
    
    print_status "Building DeepStream container for $platform platform..."
    
    docker build -f $dockerfile -t deepstream-detection$tag_suffix:latest .
    
    if [ $? -eq 0 ]; then
        print_success "Container built successfully: deepstream-detection$tag_suffix:latest"
    else
        print_error "Container build failed"
        exit 1
    fi
}

# Run function
run_container() {
    local platform=$1
    local tag_suffix=""
    
    if [ "$platform" = "jetson" ]; then
        tag_suffix="-jetson"
    fi
    
    print_status "Starting DeepStream detection system..."
    
    # Check if models directory exists
    if [ ! -d "./models" ]; then
        print_warning "Models directory not found. Creating empty directory."
        mkdir -p ./models
    fi
    
    # Check if data directory exists
    if [ ! -d "./data" ]; then
        mkdir -p ./data/frames
    fi
    
    # Check if logs directory exists
    if [ ! -d "./logs" ]; then
        mkdir -p ./logs
    fi
    
    # Run with Docker Compose for full stack
    if [ -f "docker-compose.yml" ]; then
        print_status "Using Docker Compose for full stack deployment..."
        docker compose up -d
        print_success "Full stack started. Access RTSP server web UI at http://localhost:8888"
    else
        # Run standalone container
        docker run -it --rm \
            --gpus all \
            -p 8555:8554 \
            -v $(pwd)/models:/app/models:ro \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/logs:/app/logs \
            -v $(pwd)/configs:/app/configs \
            deepstream-detection$tag_suffix:latest
    fi
}

# Stop function
stop_containers() {
    print_status "Stopping containers..."
    
    if [ -f "docker-compose.yml" ]; then
        docker compose down
    fi
    
    # Stop any running deepstream containers
    docker ps -q --filter "ancestor=deepstream-detection" | xargs -r docker stop
    docker ps -q --filter "ancestor=deepstream-detection-jetson" | xargs -r docker stop
    
    print_success "Containers stopped"
}

# Clean function
clean_containers() {
    print_status "Cleaning up containers and images..."
    
    stop_containers
    
    # Remove containers
    docker ps -aq --filter "ancestor=deepstream-detection" | xargs -r docker rm
    docker ps -aq --filter "ancestor=deepstream-detection-jetson" | xargs -r docker rm
    
    # Remove images
    docker rmi deepstream-detection:latest 2>/dev/null || true
    docker rmi deepstream-detection-jetson:latest 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build [platform]    Build container (platform: desktop|jetson|auto)"
    echo "  run [platform]      Run container (platform: desktop|jetson|auto)"
    echo "  stop                Stop running containers"
    echo "  clean               Clean up containers and images"
    echo "  logs                Show container logs"
    echo "  shell               Open shell in running container"
    echo ""
    echo "Examples:"
    echo "  $0 build auto       # Auto-detect platform and build"
    echo "  $0 run desktop      # Run on desktop GPU"
    echo "  $0 run jetson       # Run on Jetson device"
    echo "  $0 stop             # Stop all containers"
}

# Show logs
show_logs() {
    if [ -f "docker-compose.yml" ]; then
        docker compose logs -f deepstream-app
    else
        docker logs -f $(docker ps -q --filter "ancestor=deepstream-detection" | head -1) 2>/dev/null || \
        docker logs -f $(docker ps -q --filter "ancestor=deepstream-detection-jetson" | head -1) 2>/dev/null || \
        print_error "No running DeepStream containers found"
    fi
}

# Open shell
open_shell() {
    local container_id=$(docker ps -q --filter "ancestor=deepstream-detection" | head -1)
    if [ -z "$container_id" ]; then
        container_id=$(docker ps -q --filter "ancestor=deepstream-detection-jetson" | head -1)
    fi
    
    if [ -n "$container_id" ]; then
        docker exec -it $container_id /bin/bash
    else
        print_error "No running DeepStream containers found"
    fi
}

# Main script
main() {
    local command=${1:-help}
    local platform=${2:-auto}
    
    # Auto-detect platform if requested
    if [ "$platform" = "auto" ]; then
        platform=$(detect_platform)
        print_status "Auto-detected platform: $platform"
    fi
    
    case $command in
        build)
            check_docker
            build_container $platform
            ;;
        run)
            check_docker
            check_nvidia_docker
            run_container $platform
            ;;
        stop)
            check_docker
            stop_containers
            ;;
        clean)
            check_docker
            clean_containers
            ;;
        logs)
            check_docker
            show_logs
            ;;
        shell)
            check_docker
            open_shell
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"