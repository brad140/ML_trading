#!/bin/bash

# ML Trading System Deployment Script
# This script deploys the complete ML trading system with all components

set -e

echo "🚀 Starting ML Trading System Deployment..."

# Configuration
PROJECT_NAME="ml-trading-system"
AWS_REGION="us-east-1"
DOCKER_REGISTRY="your-registry.com"
KAFKA_VERSION="2.8.1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_warning "AWS CLI is not installed. AWS deployment will be skipped."
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3 first."
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install requirements
    pip install --upgrade pip
    pip install -r requirements.txt
    
    log_success "Dependencies installed"
}

# Build Docker images
build_docker_images() {
    log_info "Building Docker images..."
    
    # Build main application image
    docker build -t ${PROJECT_NAME}-app:latest .
    
    # Build ML model image
    docker build -t ${PROJECT_NAME}-ml:latest --target ml-model .
    
    # Build monitoring image
    docker build -t ${PROJECT_NAME}-monitoring:latest --target monitoring .
    
    # Build Kafka producer image
    docker build -t ${PROJECT_NAME}-producer:latest --target kafka-producer .
    
    log_success "Docker images built"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Create necessary directories
    mkdir -p data/postgres
    mkdir -p data/grafana
    mkdir -p logs
    mkdir -p models
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log_success "Docker Compose deployment completed"
}

# Check service health
check_service_health() {
    log_info "Checking service health..."
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        log_success "Services are running"
    else
        log_error "Some services failed to start"
        docker-compose logs
        exit 1
    fi
    
    # Check API endpoints
    sleep 10
    
    # Check monitoring dashboard
    if curl -f http://localhost:5000/api/health > /dev/null 2>&1; then
        log_success "Monitoring dashboard is accessible"
    else
        log_warning "Monitoring dashboard is not accessible"
    fi
    
    # Check ML model service
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "ML model service is accessible"
    else
        log_warning "ML model service is not accessible"
    fi
}

# Deploy to AWS
deploy_aws() {
    if ! command -v aws &> /dev/null; then
        log_warning "AWS CLI not found. Skipping AWS deployment."
        return
    fi
    
    log_info "Deploying to AWS..."
    
    # Create S3 buckets
    create_s3_buckets
    
    # Deploy Lambda functions
    deploy_lambda_functions
    
    # Deploy ECS services
    deploy_ecs_services
    
    log_success "AWS deployment completed"
}

# Create S3 buckets
create_s3_buckets() {
    log_info "Creating S3 buckets..."
    
    # Create buckets for models and predictions
    aws s3 mb s3://${PROJECT_NAME}-models-${AWS_REGION} --region ${AWS_REGION}
    aws s3 mb s3://${PROJECT_NAME}-predictions-${AWS_REGION} --region ${AWS_REGION}
    aws s3 mb s3://${PROJECT_NAME}-data-${AWS_REGION} --region ${AWS_REGION}
    
    log_success "S3 buckets created"
}

# Deploy Lambda functions
deploy_lambda_functions() {
    log_info "Deploying Lambda functions..."
    
    # Create deployment package
    zip -r lambda-deployment.zip lambda_handler.py src/ requirements.txt
    
    # Deploy prediction Lambda
    aws lambda create-function \
        --function-name ${PROJECT_NAME}-predictor \
        --runtime python3.9 \
        --role arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role \
        --handler lambda_handler.lambda_handler \
        --zip-file fileb://lambda-deployment.zip \
        --timeout 300 \
        --memory-size 1024 \
        --region ${AWS_REGION}
    
    # Deploy batch processing Lambda
    aws lambda create-function \
        --function-name ${PROJECT_NAME}-batch-processor \
        --runtime python3.9 \
        --role arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role \
        --handler lambda_handler.batch_predict \
        --zip-file fileb://lambda-deployment.zip \
        --timeout 900 \
        --memory-size 2048 \
        --region ${AWS_REGION}
    
    log_success "Lambda functions deployed"
}

# Deploy ECS services
deploy_ecs_services() {
    log_info "Deploying ECS services..."
    
    # Create ECS cluster
    aws ecs create-cluster \
        --cluster-name ${PROJECT_NAME}-cluster \
        --region ${AWS_REGION}
    
    # Create task definition
    cat > task-definition.json << EOF
{
    "family": "${PROJECT_NAME}-task",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "${PROJECT_NAME}-app",
            "image": "${DOCKER_REGISTRY}/${PROJECT_NAME}-app:latest",
            "portMappings": [
                {
                    "containerPort": 5000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "AWS_REGION",
                    "value": "${AWS_REGION}"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/${PROJECT_NAME}",
                    "awslogs-region": "${AWS_REGION}",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
EOF
    
    # Register task definition
    aws ecs register-task-definition \
        --cli-input-json file://task-definition.json \
        --region ${AWS_REGION}
    
    log_success "ECS services deployed"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run unit tests
    python -m pytest tests/ -v
    
    # Run integration tests
    python -m pytest tests/integration/ -v
    
    log_success "Tests completed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create Grafana dashboards
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    
    # Copy dashboard configurations
    cp monitoring/grafana/*.json monitoring/grafana/dashboards/
    cp monitoring/grafana/*.yaml monitoring/grafana/datasources/
    
    # Setup Prometheus configuration
    mkdir -p monitoring/prometheus
    cat > monitoring/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-trading-system'
    static_configs:
      - targets: ['monitoring:9090']
EOF
    
    log_success "Monitoring setup completed"
}

# Main deployment function
main() {
    log_info "Starting ML Trading System deployment..."
    
    # Parse command line arguments
    DEPLOYMENT_TYPE=${1:-"local"}
    
    case $DEPLOYMENT_TYPE in
        "local")
            log_info "Deploying locally with Docker Compose"
            check_prerequisites
            install_dependencies
            build_docker_images
            setup_monitoring
            deploy_docker_compose
            ;;
        "aws")
            log_info "Deploying to AWS"
            check_prerequisites
            install_dependencies
            build_docker_images
            deploy_aws
            ;;
        "test")
            log_info "Running tests only"
            check_prerequisites
            install_dependencies
            run_tests
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            echo "Usage: $0 [local|aws|test]"
            exit 1
            ;;
    esac
    
    log_success "Deployment completed successfully!"
    
    if [ "$DEPLOYMENT_TYPE" = "local" ]; then
        echo ""
        echo "🌐 Access your services:"
        echo "   Monitoring Dashboard: http://localhost:5000"
        echo "   ML Model API: http://localhost:8000"
        echo "   Grafana: http://localhost:3000 (admin/admin)"
        echo "   Prometheus: http://localhost:9091"
        echo ""
        echo "📊 To view logs: docker-compose logs -f"
        echo "🛑 To stop: docker-compose down"
    fi
}

# Run main function
main "$@"
