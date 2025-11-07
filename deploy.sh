#!/bin/bash

set -e 

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ENVIRONMENT="${1:-production}"
PROJECT_NAME="volatility-forge"
INSTALL_DIR="/opt/${PROJECT_NAME}"
SERVICE_NAME="volatility-api"
USER="www-data"
GROUP="www-data"

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    else
        print_error "Cannot detect OS"
        exit 1
    fi
    print_info "Detected OS: $OS $VERSION"
}

install_system_dependencies() {
    print_info "Installing system dependencies..."
    
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        apt-get update
        apt-get install -y \
            python3.10 \
            python3.10-venv \
            python3-pip \
            git \
            build-essential \
            libopenblas-dev \
            nginx \
            supervisor \
            redis-server \
            postgresql \
            postgresql-contrib
    elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ]; then
        yum install -y \
            python3.10 \
            python3-pip \
            git \
            gcc \
            gcc-c++ \
            make \
            openblas-devel \
            nginx \
            supervisor \
            redis \
            postgresql-server
    else
        print_warn "Unsupported OS. Please install dependencies manually."
    fi
}

install_poetry() {
    print_info "Installing Poetry..."
    
    if ! command -v poetry &> /dev/null; then
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="/root/.local/bin:$PATH"
    else
        print_info "Poetry already installed"
    fi
}

setup_project_directory() {
    print_info "Setting up project directory..."
    
    # Create directory if it doesn't exist
    mkdir -p "$INSTALL_DIR"
    
    # Copy project files
    if [ -d ".git" ]; then
        print_info "Cloning from git repository..."
        git clone . "$INSTALL_DIR"
    else
        print_info "Copying files..."
        cp -r . "$INSTALL_DIR/"
    fi
    
    cd "$INSTALL_DIR"
}

setup_python_environment() {
    print_info "Setting up Python environment..."
    
    # Create virtual environment
    python3.10 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    if [ -f "pyproject.toml" ]; then
        print_info "Installing with Poetry..."
        poetry config virtualenvs.create false
        poetry install --only main
    elif [ -f "requirements.txt" ]; then
        print_info "Installing with pip..."
        pip install -r requirements.txt
    else
        print_error "No dependency file found"
        exit 1
    fi
}

setup_configuration() {
    print_info "Setting up configuration..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOF
VOLATILITY_ENV=${ENVIRONMENT}
MODEL_REGISTRY_PATH=${INSTALL_DIR}/model_registry
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=INFO
ENABLE_METRICS=true
EOF
    fi
    
    # Create necessary directories
    mkdir -p logs checkpoints outputs model_registry .cache
    
    # Set permissions
    chown -R "$USER:$GROUP" "$INSTALL_DIR"
}

setup_systemd_service() {
    print_info "Setting up systemd service..."
    
    cat > "/etc/systemd/system/${SERVICE_NAME}.service" << EOF
[Unit]
Description=Volatility Forge API
After=network.target redis.service postgresql.service

[Service]
Type=notify
User=${USER}
Group=${GROUP}
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=${INSTALL_DIR}/venv/bin"
Environment="VOLATILITY_ENV=${ENVIRONMENT}"
ExecStart=${INSTALL_DIR}/venv/bin/gunicorn api.main:app \\
    --workers 4 \\
    --worker-class uvicorn.workers.UvicornWorker \\
    --bind 0.0.0.0:8000 \\
    --timeout 120 \\
    --access-logfile ${INSTALL_DIR}/logs/access.log \\
    --error-logfile ${INSTALL_DIR}/logs/error.log
Restart=always
RestartSec=10
StandardOutput=append:${INSTALL_DIR}/logs/service.log
StandardError=append:${INSTALL_DIR}/logs/service-error.log

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
}

setup_nginx() {
    print_info "Setting up Nginx reverse proxy..."
    
    cat > "/etc/nginx/sites-available/${PROJECT_NAME}" << 'EOF'
server {
    listen 80;
    server_name _;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
    
    location /metrics {
        proxy_pass http://127.0.0.1:8000/metrics;
        allow 127.0.0.1;
        deny all;
    }
}
EOF
    
    # Enable site
    ln -sf "/etc/nginx/sites-available/${PROJECT_NAME}" "/etc/nginx/sites-enabled/"
    
    # Test configuration
    nginx -t
    
    systemctl reload nginx
}

setup_redis() {
    print_info "Configuring Redis..."
    
    sed -i 's/# maxmemory <bytes>/maxmemory 2gb/' /etc/redis/redis.conf
    sed -i 's/# maxmemory-policy noeviction/maxmemory-policy allkeys-lru/' /etc/redis/redis.conf
    
    systemctl enable redis-server
    systemctl start redis-server
}

setup_postgresql() {
    print_info "Configuring PostgreSQL..."
    
    if [ "$OS" = "centos" ] || [ "$OS" = "rhel" ]; then
        postgresql-setup --initdb
    fi
    
    systemctl enable postgresql
    systemctl start postgresql
    
    sudo -u postgres psql << EOF
CREATE DATABASE volatility_db;
CREATE USER volatility_user WITH ENCRYPTED PASSWORD 'changeme';
GRANT ALL PRIVILEGES ON DATABASE volatility_db TO volatility_user;
EOF
}

setup_monitoring() {
    print_info "Setting up monitoring..."
    
    if command -v prometheus &> /dev/null; then
        cat >> /etc/prometheus/prometheus.yml << EOF

  - job_name: 'volatility-forge'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
EOF
        systemctl reload prometheus
    fi
}

start_services() {
    print_info "Starting services..."
    
    systemctl start "$SERVICE_NAME"
    systemctl status "$SERVICE_NAME"
}

verify_deployment() {
    print_info "Verifying deployment..."
    
    sleep 5
    
    # Check if API is responding
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_info "API is responding successfully!"
    else
        print_error "API health check failed"
        journalctl -u "$SERVICE_NAME" -n 50
        exit 1
    fi
}

print_summary() {
    print_info "Deployment complete!"
    echo ""
    echo "====================="
    echo "Deployment Summary"
    echo "====================="
    echo "Environment: $ENVIRONMENT"
    echo "Installation Directory: $INSTALL_DIR"
    echo "Service Name: $SERVICE_NAME"
    echo ""
    echo "Useful Commands:"
    echo "  - View logs: sudo journalctl -u $SERVICE_NAME -f"
    echo "  - Restart service: sudo systemctl restart $SERVICE_NAME"
    echo "  - Check status: sudo systemctl status $SERVICE_NAME"
    echo ""
    echo "API Endpoints:"
    echo "  - Documentation: http://localhost/docs"
    echo "  - Health Check: http://localhost/health"
    echo "  - Metrics: http://localhost/metrics"
    echo ""
}

main() {
    print_info "Starting deployment for environment: $ENVIRONMENT"
    
    check_root
    detect_os
    install_system_dependencies
    install_poetry
    setup_project_directory
    setup_python_environment
    setup_configuration
    setup_systemd_service
    setup_nginx
    setup_redis
    # setup_postgresql  # Uncomment if using PostgreSQL
    setup_monitoring
    start_services
    verify_deployment
    print_summary
}

main "$@"