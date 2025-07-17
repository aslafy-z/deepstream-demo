#!/bin/bash

# RTSP Object Detection System - Complete Startup Script
# This script starts the entire system with RTSP flow, FFmpeg, and DeepStream

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

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

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check NVIDIA Docker runtime
    if docker info 2>/dev/null | grep -q nvidia; then
        print_success "NVIDIA Docker runtime detected"
    else
        print_warning "NVIDIA Docker runtime not detected. GPU acceleration may not work."
    fi
    
    print_success "Prerequisites check completed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p models data/frames logs test-videos
    
    # Create sample input video if it doesn't exist
    if [ ! -f "videos/input.mp4" ]; then
        print_status "Creating sample input video..."
        if command -v ffmpeg &> /dev/null; then
            ffmpeg -f lavfi -i testsrc2=duration=60:size=1280x720:rate=30 \
                   -f lavfi -i sine=frequency=1000:duration=60 \
                   -c:v libx264 -preset ultrafast -c:a aac -shortest \
                   -y videos/input.mp4 2>/dev/null || print_warning "Could not create sample video"
        else
            print_warning "FFmpeg not found. Please provide a videos/input.mp4 file."
        fi
    fi
    
    print_success "Directories created"
}

# Start the system
start_system() {
    local compose_file=${1:-docker-compose.yml}
    
    print_header "Starting RTSP Object Detection System"
    print_status "Using compose file: $compose_file"
    
    # Pull latest images
    print_status "Pulling latest Docker images..."
    docker-compose -f $compose_file pull
    
    # Build custom images
    print_status "Building custom images..."
    docker-compose -f $compose_file build
    
    # Start services
    print_status "Starting services..."
    docker-compose -f $compose_file up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to start..."
    sleep 10
    
    # Check service health
    check_services_health $compose_file
}

# Check service health
check_services_health() {
    local compose_file=$1
    
    print_status "Checking service health..."
    
    # Get running containers
    local containers=$(docker-compose -f $compose_file ps --services)
    
    for service in $containers; do
        local container_name=$(docker-compose -f $compose_file ps -q $service)
        if [ -n "$container_name" ]; then
            local status=$(docker inspect --format='{{.State.Status}}' $container_name 2>/dev/null || echo "not found")
            if [ "$status" = "running" ]; then
                print_success "$service: Running"
            else
                print_warning "$service: $status"
            fi
        else
            print_warning "$service: Not found"
        fi
    done
}

# Show system information
show_system_info() {
    print_header "System Information"
    
    echo -e "${CYAN}Access Points:${NC}"
    echo "  🎥 RTSP Input Stream:     rtsp://localhost:8554/live.stream"
    echo "  🔍 Detection Output:      rtsp://localhost:8555/detection.stream"
    echo "  🌐 Web Dashboard:         http://localhost:8080"
    echo "  📺 MediaMTX Web UI:       http://localhost:8888"
    echo "  📊 HLS Input Stream:      http://localhost:8888/live.stream/"
    echo "  � HLS Output Stream:     http://localhost:8888/detection.stream/"
    echo "  🔌 MQTT Broker:           mqtt://localhost:1883"
    echo ""
    
    echo -e "${CYAN}Test Commands:${NC}"
    echo "  # View input stream with VLC or FFplay:"
    echo "  ffplay rtsp://localhost:8554/live.stream"
    echo ""
    echo "  # View detection output:"
    echo "  ffplay rtsp://localhost:8555/detection.stream"
    echo ""
    echo "  # Subscribe to MQTT events:"
    echo "  mosquitto_sub -h localhost -t 'detection/events'"
    echo ""
    echo "  # Send test HTTP event:"
    echo "  curl -X POST http://localhost:8090/events -H 'Content-Type: application/json' -d '{\"test\": \"event\"}'"
    echo ""
    
    echo -e "${CYAN}Management Commands:${NC}"
    echo "  # View logs:"
    echo "  docker-compose logs -f deepstream-app"
    echo ""
    echo "  # Stop system:"
    echo "  docker-compose down"
    echo ""
    echo "  # Restart specific service:"
    echo "  docker-compose restart deepstream-app"
    echo ""
}

# Show logs
show_logs() {
    local service=${1:-deepstream-app}
    local compose_file=${2:-docker-compose.yml}
    
    print_status "Showing logs for $service..."
    docker-compose -f $compose_file logs -f $service
}

# Stop the system
stop_system() {
    local compose_file=${1:-docker-compose.yml}
    
    print_header "Stopping RTSP Object Detection System"
    docker-compose -f $compose_file down
    print_success "System stopped"
}

# Clean up everything
cleanup_system() {
    local compose_file=${1:-docker-compose.yml}
    
    print_header "Cleaning up RTSP Object Detection System"
    
    # Stop and remove containers
    docker-compose -f $compose_file down -v --remove-orphans
    
    # Remove custom images
    docker rmi $(docker images | grep deepstream-detection | awk '{print $3}') 2>/dev/null || true
    
    # Clean up unused Docker resources
    docker system prune -f
    
    print_success "Cleanup completed"
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start [compose-file]     Start the complete system (default: docker-compose.yml)"
    echo "  start-dev               Start development environment (docker-compose.dev.yml)"
    echo "  stop [compose-file]     Stop the system"
    echo "  restart [compose-file]  Restart the system"
    echo "  logs [service]          Show logs for a service (default: deepstream-app)"
    echo "  status [compose-file]   Show system status"
    echo "  info                    Show system information and access points"
    echo "  cleanup [compose-file]  Stop system and clean up resources"
    echo "  health                  Check system health"
    echo ""
    echo "Examples:"
    echo "  $0 start                # Start with default configuration"
    echo "  $0 start-dev            # Start development environment"
    echo "  $0 logs deepstream-app  # Show DeepStream application logs"
    echo "  $0 status               # Check system status"
    echo "  $0 info                 # Show access points and commands"
}

# Main script
main() {
    local command=${1:-help}
    local arg1=${2:-}
    local arg2=${3:-}
    
    case $command in
        start)
            check_prerequisites
            create_directories
            start_system ${arg1:-docker-compose.yml}
            show_system_info
            ;;
        start-dev)
            check_prerequisites
            create_directories
            start_system docker-compose.dev.yml
            show_system_info
            ;;
        stop)
            stop_system ${arg1:-docker-compose.yml}
            ;;
        restart)
            stop_system ${arg1:-docker-compose.yml}
            sleep 2
            start_system ${arg1:-docker-compose.yml}
            ;;
        logs)
            show_logs ${arg1:-deepstream-app} ${arg2:-docker-compose.yml}
            ;;
        status)
            check_services_health ${arg1:-docker-compose.yml}
            ;;
        info)
            show_system_info
            ;;
        cleanup)
            cleanup_system ${arg1:-docker-compose.yml}
            ;;
        health)
            check_prerequisites
            check_services_health ${arg1:-docker-compose.yml}
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