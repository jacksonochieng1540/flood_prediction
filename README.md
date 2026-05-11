# Flood Prediction System

##  Overview

The Flood Prediction System is a Django-based web application that uses machine learning to predict flood risks in real-time. The system analyzes various environmental factors to provide early warnings and risk assessments for flood-prone areas.

### Key Capabilities

- **Real-time Prediction**: Instant flood risk assessment using ML models
- **Historical Analysis**: Track and analyze past predictions and outcomes
- **Alert System**: Automated alerts for high-risk scenarios
- **Analytics Dashboard**: Comprehensive data visualization and insights
- **Multi-user Support**: Role-based access control and user management

## Features

### Prediction & Analysis
- **Real-time Flood Prediction** using ensemble ML models
- **Feature Importance Analysis** to understand key risk factors
- **Batch Prediction** for multiple locations simultaneously
- **Historical Data Tracking** with outcome verification

###  Alert Management
- **Automated Alert Generation** for high-risk predictions
- **Multi-level Severity** (Low, Moderate, High, Critical)
- **Alert Resolution Tracking** with user accountability
- **Real-time Notification System**

### Dashboard & Analytics
- **Interactive Dashboard** with key metrics and statistics
- **Risk Distribution Charts** and trend analysis
- **Model Performance Monitoring**
- **Export Capabilities** (CSV, Reports)

### System Features
- **User Authentication & Authorization**
- **RESTful API** for integration
- **Responsive Design** for mobile and desktop
- **Admin Interface** for system management
- **Health Monitoring** and system status checks


##  Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Redis (for caching and Celery)

### 5-Minute Setup

```bash
# Clone the repository
git clone https://github.com/jackdonochieng1540/flood-prediction-system.git
cd flood-prediction-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Train the ML model
python manage.py train_model
#start the django server
Next you can stary the development  server 
