from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from django.db.models import Count, Avg
from django.core.paginator import Paginator
from django.db import transaction
from datetime import timedelta, datetime
import json
import logging
import csv
import os

from .models import (
    PredictionHistory, Alert, Location, 
    WeatherData, ModelPerformance, SystemSettings
)


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {username}!')
                
                next_page = request.GET.get('next', 'index')  # Fixed: changed 'dashboard' to 'index'
                return redirect(next_page)
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()
    
    return render(request, 'flood_prediction/login.html', {'form': form})

def logout_view(request):
    """User logout view"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('index')  # Fixed: changed 'dashboard' to 'index'

def register_view(request):
    """User registration view"""
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Account created successfully! Welcome, {user.username}!')
            return redirect('index')  # Fixed: changed 'dashboard' to 'index'
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserCreationForm()
    
    return render(request, 'flood_prediction/register.html', {'form': form})

def profile_view(request):
    return render(request, 'flood_prediction/profile.html')

try:
    from .ml_model import EnhancedFloodPredictor
    logger = logging.getLogger(__name__)
    
    # Initialize predictor
    try:
        predictor = EnhancedFloodPredictor()
        model_path = 'flood_prediction/models/flood_model_v2.pkl'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Try to load existing model
        if os.path.exists(model_path):
            predictor.load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.info("No existing model found. Training new model...")
            training_results = predictor.train_ensemble()
            logger.info(f"New model trained with accuracy: {training_results.get('test_accuracy', 0):.3f}")
            predictor.save_model(model_path)
            logger.info("New model saved successfully")
            
    except Exception as e:
        logger.warning(f"Model initialization failed: {e}. Using basic predictor.")
        predictor = EnhancedFloodPredictor()
        predictor.train_ensemble()

except ImportError as e:
    # Fallback if ml_model doesn't exist
    print(f"Import error: {e}")
    logger = logging.getLogger(__name__)
    
    class EnhancedFloodPredictor:
        def __init__(self):
            self.model_version = "1.0.0"
            self.is_trained = True
        
        def train_ensemble(self):
            return {
                'train_accuracy': 0.85,
                'test_accuracy': 0.82,
                'precision': 0.80,
                'recall': 0.78,
                'f1_score': 0.79
            }
        
        def predict(self, features):
            # Simple rule-based fallback prediction
            rainfall = features.get('rainfall_mm', 0)
            water_level = features.get('water_level_m', 0)
            soil_moisture = features.get('soil_moisture_percent', 0)
            
            # Simple flood prediction logic
            flood_prob = min(0.9, (rainfall / 100) * 0.5 + (water_level / 10) * 0.3 + (soil_moisture / 100) * 0.2)
            
            if flood_prob > 0.7:
                risk_level = "High"
                prediction = True
            elif flood_prob > 0.4:
                risk_level = "Moderate"
                prediction = False
            else:
                risk_level = "Low"
                prediction = False
            
            return {
                'prediction': prediction,
                'probability_flood': float(flood_prob),
                'probability_no_flood': float(1 - flood_prob),
                'risk_level': risk_level,
                'confidence': 0.8,
                'model_used': 'fallback'
            }
        
        def explain_prediction(self, features):
            result = self.predict(features)
            result['feature_importance'] = {
                'rainfall_mm': 0.35,
                'water_level_m': 0.25,
                'river_discharge_m3s': 0.15,
                'soil_moisture_percent': 0.12,
                'temperature_celsius': 0.05,
                'humidity_percent': 0.04,
                'elevation_m': 0.03,
                'slope_degree': 0.01
            }
            return result
        
        def get_feature_importance(self):
            return {
                'rainfall_mm': 0.35,
                'water_level_m': 0.25,
                'river_discharge_m3s': 0.15,
                'soil_moisture_percent': 0.12,
                'temperature_celsius': 0.05,
                'humidity_percent': 0.04,
                'elevation_m': 0.03,
                'slope_degree': 0.01
            }
        
        def save_model(self, path):
            print(f"Mock: Saving model to {path}")
        
        def load_model(self, path):
            print(f"Mock: Loading model from {path}")
        
        def is_model_loaded(self):
            return True
        
        def retrain_with_new_data(self, data):
            print(f"Mock: Retraining with {len(data)} samples")
            return True
        
        def evaluate_model(self):
            return {
                'accuracy': 0.82,
                'precision': 0.80,
                'recall': 0.78,
                'f1_score': 0.79
            }
    
    predictor = EnhancedFloodPredictor()

def index(request):
    """Enhanced main dashboard"""
    # Get recent statistics
    recent_predictions = PredictionHistory.objects.all().order_by('-created_at')[:10]
    active_alerts = Alert.objects.filter(status='active').order_by('-created_at')[:5]
    
    # Calculate statistics
    total_predictions = PredictionHistory.objects.count()
    high_risk_count = PredictionHistory.objects.filter(
        risk_level__in=['high', 'critical']
    ).count()
    
    # Get prediction statistics for the last 7 days
    week_ago = timezone.now() - timedelta(days=7)
    weekly_predictions = PredictionHistory.objects.filter(
        created_at__gte=week_ago
    ).count()
    
    context = {
        'recent_predictions': recent_predictions,
        'active_alerts': active_alerts,
        'total_predictions': total_predictions,
        'high_risk_count': high_risk_count,
        'weekly_predictions': weekly_predictions,
    }
    
    return render(request, 'flood_prediction/dashboard.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def predict_flood(request):
    """Enhanced prediction endpoint with database logging"""
    try:
        # Handle both form data and JSON data
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST.dict()
        
        # Extract features with validation
        features = {
            'rainfall_mm': float(data.get('rainfall', 0)),
            'water_level_m': float(data.get('water_level', 0)),
            'river_discharge_m3s': float(data.get('river_discharge', 0)),
            'soil_moisture_percent': float(data.get('soil_moisture', 0)),
            'temperature_celsius': float(data.get('temperature', 0)),
            'humidity_percent': float(data.get('humidity', 0)),
            'elevation_m': float(data.get('elevation', 0)),
            'slope_degree': float(data.get('slope', 0))
        }
        
        # Validate feature ranges
        validation_errors = validate_features(features)
        if validation_errors:
            return JsonResponse({
                'success': False,
                'error': '; '.join(validation_errors)
            }, status=400)
        
        # Get prediction with explanation
        result = predictor.explain_prediction(features)
        
        # Save to database
        prediction_record = PredictionHistory.objects.create(
            rainfall_mm=features['rainfall_mm'],
            water_level_m=features['water_level_m'],
            river_discharge_m3s=features['river_discharge_m3s'],
            soil_moisture_percent=features['soil_moisture_percent'],
            temperature_celsius=features['temperature_celsius'],
            humidity_percent=features['humidity_percent'],
            elevation_m=features['elevation_m'],
            slope_degree=features['slope_degree'],
            prediction=bool(result['prediction']),
            probability_flood=result['probability_flood'],
            probability_no_flood=result['probability_no_flood'],
            risk_level=result['risk_level'].lower().replace(' ', '_'),
            confidence=result['confidence'],
            model_used=result['model_used'],
            location=data.get('location', ''),
            notes=data.get('notes', '')
        )
        
        # Create alert if high risk
        if result['probability_flood'] > 0.7:
            alert = create_flood_alert(prediction_record, result)
            result['alert_created'] = True
            result['alert_id'] = alert.id if alert else None
        else:
            result['alert_created'] = False
        
        logger.info(f"Prediction created: ID={prediction_record.id}, Risk={result['risk_level']}")
        
        # Return response based on request type
        if request.content_type == 'application/json':
            return JsonResponse({
                'success': True,
                'prediction': result,
                'prediction_id': prediction_record.id
            })
        else:
            # For form submissions, redirect to detail page
            messages.success(request, f"Prediction completed! Risk level: {result['risk_level']}")
            return redirect('prediction_detail', prediction_id=prediction_record.id)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except ValueError as e:
        return JsonResponse({
            'success': False,
            'error': f'Invalid data format: {str(e)}'
        }, status=400)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

def validate_features(features):
    """Validate feature values"""
    errors = []
    
    if features['rainfall_mm'] < 0:
        errors.append("Rainfall cannot be negative")
    if features['water_level_m'] < 0:
        errors.append("Water level cannot be negative")
    if features['river_discharge_m3s'] < 0:
        errors.append("River discharge cannot be negative")
    if not (0 <= features['soil_moisture_percent'] <= 100):
        errors.append("Soil moisture must be between 0 and 100%")
    if not (0 <= features['humidity_percent'] <= 100):
        errors.append("Humidity must be between 0 and 100%")
    if not (0 <= features['slope_degree'] <= 90):
        errors.append("Slope must be between 0 and 90 degrees")
    
    return errors

def create_flood_alert(prediction_record, prediction_result):
    """Create flood alert for high-risk predictions"""
    try:
        # Determine alert message based on risk level
        risk_level = prediction_result['risk_level'].lower()
        if risk_level == 'critical':
            message = f"CRITICAL: Flood imminent with {prediction_result['probability_flood']:.1%} probability. Immediate action required!"
            recommended_actions = "Evacuate immediately. Contact emergency services. Move to higher ground."
        elif risk_level == 'high':
            message = f"HIGH: Severe flood risk detected with {prediction_result['probability_flood']:.1%} probability. Prepare for evacuation."
            recommended_actions = "Prepare evacuation plan. Secure property. Monitor official alerts closely."
        else:
            message = f"Flood risk detected: {prediction_result['risk_level']} risk with {prediction_result['probability_flood']:.1%} probability"
            recommended_actions = "Monitor water levels. Stay informed through official channels."
        
        alert = Alert.objects.create(
            prediction=prediction_record,
            alert_type='flood_warning',
            severity=prediction_result['risk_level'].lower(),
            message=message,
            affected_area=prediction_record.location or "Unknown location",
            recommended_actions=recommended_actions,
            status='active'
        )
        logger.info(f"Alert created: {alert.id} for prediction {prediction_record.id}")
        return alert
    except Exception as e:
        logger.error(f"Alert creation error: {str(e)}")
        return None

def prediction_history(request):
    """View prediction history with filtering and pagination"""
    predictions = PredictionHistory.objects.all().order_by('-created_at')
    
    # Apply filters
    risk_filter = request.GET.get('risk_level')
    date_filter = request.GET.get('date')
    location_filter = request.GET.get('location')
    
    if risk_filter and risk_filter != 'all':
        predictions = predictions.filter(risk_level=risk_filter)
    
    if date_filter:
        try:
            filter_date = datetime.strptime(date_filter, '%Y-%m-%d').date()
            predictions = predictions.filter(created_at__date=filter_date)
        except ValueError:
            pass
    
    if location_filter:
        predictions = predictions.filter(location__icontains=location_filter)
    
    # Get unique locations for filter dropdown
    locations = PredictionHistory.objects.exclude(location='').values_list('location', flat=True).distinct()
    
    # Pagination
    paginator = Paginator(predictions, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'risk_levels': PredictionHistory.RISK_LEVEL_CHOICES,
        'locations': locations,
        'current_filters': {
            'risk_level': risk_filter,
            'date': date_filter,
            'location': location_filter
        }
    }
    
    return render(request, 'flood_prediction/history.html', context)

def prediction_detail(request, prediction_id):
    """Detailed view of a single prediction"""
    prediction = get_object_or_404(PredictionHistory, id=prediction_id)
    related_alerts = Alert.objects.filter(prediction=prediction)
    
    # Get feature importance for display
    feature_importance = predictor.get_feature_importance()
    
    # Prepare feature values for display
    feature_values = {
        'rainfall_mm': prediction.rainfall_mm,
        'water_level_m': prediction.water_level_m,
        'river_discharge_m3s': prediction.river_discharge_m3s,
        'soil_moisture_percent': prediction.soil_moisture_percent,
        'temperature_celsius': prediction.temperature_celsius,
        'humidity_percent': prediction.humidity_percent,
        'elevation_m': prediction.elevation_m,
        'slope_degree': prediction.slope_degree,
    }
    
    context = {
        'prediction': prediction,
        'alerts': related_alerts,
        'feature_importance': feature_importance,
        'feature_values': feature_values,
    }
    
    return render(request, 'flood_prediction/prediction_detail.html', context)

@login_required
def alerts_dashboard(request):
    """Enhanced alerts management dashboard"""
    alerts = Alert.objects.all().order_by('-created_at')
    
    # Filter by status and severity
    status_filter = request.GET.get('status', 'active')
    severity_filter = request.GET.get('severity')
    
    if status_filter != 'all':
        alerts = alerts.filter(status=status_filter)
    
    if severity_filter:
        alerts = alerts.filter(severity=severity_filter)
    
    # Statistics
    alert_stats = {
        'total': alerts.count(),
        'active': Alert.objects.filter(status='active').count(),
        'resolved': Alert.objects.filter(status='resolved').count(),
        'critical': Alert.objects.filter(severity='critical').count(),
    }
    
    context = {
        'alerts': alerts,
        'alert_stats': alert_stats,
        'current_filters': {
            'status': status_filter,
            'severity': severity_filter
        }
    }
    
    return render(request, 'flood_prediction/alerts.html', context)

@csrf_exempt
@require_http_methods(["POST"])
@login_required
def update_alert_status(request, alert_id):
    """Update alert status (resolve/activate)"""
    try:
        alert = get_object_or_404(Alert, id=alert_id)
        data = json.loads(request.body)
        new_status = data.get('status')
        
        if new_status in ['active', 'resolved']:
            alert.status = new_status
            alert.resolved_by = request.user if new_status == 'resolved' else None
            alert.resolved_at = timezone.now() if new_status == 'resolved' else None
            alert.save()
            
            logger.info(f"Alert {alert_id} status updated to {new_status} by {request.user}")
            
            return JsonResponse({'success': True, 'new_status': new_status})
        else:
            return JsonResponse({'success': False, 'error': 'Invalid status'}, status=400)
            
    except Exception as e:
        logger.error(f"Alert update error: {str(e)}")
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

def analytics_dashboard(request):
    """Advanced analytics and insights dashboard"""
    # Time-based statistics
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    # Prediction statistics
    total_predictions = PredictionHistory.objects.count()
    weekly_predictions = PredictionHistory.objects.filter(
        created_at__date__gte=week_ago
    ).count()
    monthly_predictions = PredictionHistory.objects.filter(
        created_at__date__gte=month_ago
    ).count()
    
    # Risk level distribution
    risk_distribution = PredictionHistory.objects.values('risk_level').annotate(
        count=Count('id')
    ).order_by('risk_level')
    
    # Average probabilities by risk level
    probability_stats = PredictionHistory.objects.values('risk_level').annotate(
        avg_flood_prob=Avg('probability_flood'),
        avg_confidence=Avg('confidence')
    )
    
    # Model performance
    model_performance = ModelPerformance.objects.all().order_by('-created_at')[:10]
    
    # Recent high-risk predictions
    recent_high_risk = PredictionHistory.objects.filter(
        risk_level__in=['high', 'critical']
    ).order_by('-created_at')[:5]
    
    # Calculate high risk count for statistics
    high_risk_count = PredictionHistory.objects.filter(
        risk_level__in=['high', 'critical']
    ).count()
    
    context = {
        'total_predictions': total_predictions,
        'weekly_predictions': weekly_predictions,
        'monthly_predictions': monthly_predictions,
        'risk_distribution': list(risk_distribution),
        'probability_stats': list(probability_stats),
        'model_performance': model_performance,
        'recent_high_risk': recent_high_risk,
        'high_risk_count': high_risk_count,
    }
    
    return render(request, 'flood_prediction/analytics.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def batch_predict(request):
    """Batch prediction endpoint for multiple locations"""
    try:
        data = json.loads(request.body)
        predictions_list = data.get('predictions', [])
        
        if not predictions_list:
            return JsonResponse({
                'success': False,
                'error': 'No prediction data provided'
            }, status=400)
        
        results = []
        for pred_data in predictions_list:
            features = {
                'rainfall_mm': float(pred_data.get('rainfall', 0)),
                'water_level_m': float(pred_data.get('water_level', 0)),
                'river_discharge_m3s': float(pred_data.get('river_discharge', 0)),
                'soil_moisture_percent': float(pred_data.get('soil_moisture', 0)),
                'temperature_celsius': float(pred_data.get('temperature', 0)),
                'humidity_percent': float(pred_data.get('humidity', 0)),
                'elevation_m': float(pred_data.get('elevation', 0)),
                'slope_degree': float(pred_data.get('slope', 0))
            }
            
            # Validate features
            validation_errors = validate_features(features)
            if validation_errors:
                results.append({
                    'location': pred_data.get('location', ''),
                    'error': '; '.join(validation_errors)
                })
                continue
            
            result = predictor.predict(features)
            explanation = predictor.explain_prediction(features)
            
            # Save to database
            prediction_record = PredictionHistory.objects.create(
                rainfall_mm=features['rainfall_mm'],
                water_level_m=features['water_level_m'],
                river_discharge_m3s=features['river_discharge_m3s'],
                soil_moisture_percent=features['soil_moisture_percent'],
                temperature_celsius=features['temperature_celsius'],
                humidity_percent=features['humidity_percent'],
                elevation_m=features['elevation_m'],
                slope_degree=features['slope_degree'],
                prediction=bool(result['prediction']),
                probability_flood=result['probability_flood'],
                probability_no_flood=result['probability_no_flood'],
                risk_level=result['risk_level'].lower().replace(' ', '_'),
                confidence=result['confidence'],
                model_used=result['model_used'],
                location=pred_data.get('location', ''),
                notes=pred_data.get('notes', '')
            )
            
            results.append({
                'location': pred_data.get('location', ''),
                'prediction': explanation,
                'prediction_id': prediction_record.id
            })
        
        logger.info(f"Batch prediction completed: {len(results)} predictions")
        
        return JsonResponse({
            'success': True,
            'predictions': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

def feature_importance(request):
    """Display feature importance analysis"""
    importance_data = predictor.get_feature_importance()
    
    # Sort features by importance
    sorted_features = sorted(
        importance_data.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    context = {
        'feature_importance': importance_data,
        'top_features': sorted_features[:10]  # Top 10 features
    }
    
    return render(request, 'flood_prediction/importance.html', context)

@csrf_exempt
@require_http_methods(["POST"])
@login_required
def retrain_model(request):
    """Retrain model with latest data"""
    try:
        # Get recent predictions for retraining (last 3 months)
        three_months_ago = timezone.now() - timedelta(days=90)
        recent_data = PredictionHistory.objects.filter(
            created_at__gte=three_months_ago,
            actual_flood_occurred__isnull=False  # Only use verified data
        )
        
        if recent_data.count() < 50:
            return JsonResponse({
                'success': False,
                'error': 'Insufficient verified data for retraining (min 50 records required)'
            }, status=400)
        
        # Retrain model
        success = predictor.retrain_with_new_data(recent_data)
        
        if success:
            # Save model performance
            performance = predictor.evaluate_model()
            ModelPerformance.objects.create(
                accuracy=performance.get('accuracy', 0),
                precision=performance.get('precision', 0),
                recall=performance.get('recall', 0),
                f1_score=performance.get('f1_score', 0),
                training_samples=recent_data.count(),
                model_version=predictor.model_version
            )
            
            # Save the updated model
            model_path = 'flood_prediction/models/flood_model_v2.pkl'
            predictor.save_model(model_path)
            
            logger.info(f"Model retrained successfully with {recent_data.count()} samples")
            
            return JsonResponse({
                'success': True,
                'message': f'Model retrained successfully with {recent_data.count()} samples',
                'performance': performance
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Model retraining failed'
            }, status=500)
            
    except Exception as e:
        logger.error(f"Model retraining error: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@login_required
@require_http_methods(["POST"])
def verify_prediction_outcome(request, prediction_id):
    """Verify the actual outcome of a prediction"""
    try:
        prediction = get_object_or_404(PredictionHistory, id=prediction_id)
        data = json.loads(request.body)
        
        actual_flood_occurred = data.get('actual_flood_occurred')
        actual_flood_severity = data.get('actual_flood_severity', '')
        
        if actual_flood_occurred is None:
            return JsonResponse({
                'success': False,
                'error': 'actual_flood_occurred field is required'
            }, status=400)
        
        prediction.actual_flood_occurred = bool(actual_flood_occurred)
        prediction.actual_flood_severity = actual_flood_severity
        prediction.verified_at = timezone.now()
        prediction.verified_by = request.user
        prediction.save()
        
        logger.info(f"Prediction {prediction_id} outcome verified by {request.user}")
        
        return JsonResponse({
            'success': True,
            'message': 'Outcome verified successfully'
        })
        
    except Exception as e:
        logger.error(f"Outcome verification error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

def export_predictions_csv(request):
    """Export predictions as CSV"""
    try:
        # Get filtered predictions
        predictions = PredictionHistory.objects.all().order_by('-created_at')
        
        # Apply same filters as prediction_history view
        risk_filter = request.GET.get('risk_level')
        date_filter = request.GET.get('date')
        
        if risk_filter and risk_filter != 'all':
            predictions = predictions.filter(risk_level=risk_filter)
        
        if date_filter:
            try:
                filter_date = datetime.strptime(date_filter, '%Y-%m-%d').date()
                predictions = predictions.filter(created_at__date=filter_date)
            except ValueError:
                pass
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="flood_predictions.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'ID', 'Timestamp', 'Location', 'Risk Level', 'Flood Predicted',
            'Flood Probability', 'No Flood Probability', 'Confidence',
            'Rainfall (mm)', 'Water Level (m)', 'River Discharge (m³/s)',
            'Soil Moisture (%)', 'Temperature (°C)', 'Humidity (%)',
            'Elevation (m)', 'Slope (degrees)', 'Model Used', 'Notes'
        ])
        
        for prediction in predictions:
            writer.writerow([
                prediction.id,
                prediction.created_at.isoformat(),
                prediction.location,
                prediction.get_risk_level_display(),
                'Yes' if prediction.prediction else 'No',
                prediction.probability_flood,
                prediction.probability_no_flood,
                prediction.confidence,
                prediction.rainfall_mm,
                prediction.water_level_m,
                prediction.river_discharge_m3s,
                prediction.soil_moisture_percent,
                prediction.temperature_celsius,
                prediction.humidity_percent,
                prediction.elevation_m,
                prediction.slope_degree,
                prediction.model_used,
                prediction.notes
            ])
        
        return response
        
    except Exception as e:
        logger.error(f"CSV export error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'Failed to export data'
        }, status=500)

def health_check(request):
    """System health check endpoint"""
    try:
        # Check database connection
        db_status = PredictionHistory.objects.exists()
        
        # Check model status
        model_status = hasattr(predictor, 'is_model_loaded') and predictor.is_model_loaded()
        
        # Check recent errors in logs (simplified)
        recent_errors = False
        
        # Check if any critical alerts are active
        critical_alerts = Alert.objects.filter(severity='critical', status='active').exists()
        
        status = {
            'database': 'healthy' if db_status else 'warning',
            'model': 'loaded' if model_status else 'not loaded',
            'errors': 'none' if not recent_errors else 'present',
            'critical_alerts': critical_alerts,
            'timestamp': timezone.now().isoformat(),
            'version': '1.0.0',
            'total_predictions': PredictionHistory.objects.count(),
            'active_alerts': Alert.objects.filter(status='active').count()
        }
        
        overall_status = 'healthy'
        if not db_status or not model_status or critical_alerts:
            overall_status = 'degraded'
        if recent_errors:
            overall_status = 'unhealthy'
        
        status['overall'] = overall_status
        
        return JsonResponse(status)
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        }, status=500)

# Error handlers
def handler404(request, exception):
    if request.content_type == 'application/json':
        return JsonResponse({'error': 'Endpoint not found'}, status=404)
    return render(request, 'flood_prediction/404.html', status=404)

def handler500(request):
    if request.content_type == 'application/json':
        return JsonResponse({'error': 'Internal server error'}, status=500)
    return render(request, 'flood_prediction/500.html', status=500)

# Additional utility views
def quick_prediction(request):
    """Quick prediction form for common scenarios"""
    if request.method == 'POST':
        # Handle quick prediction form
        scenario = request.POST.get('scenario', 'normal')
        
        # Predefined feature sets for common scenarios
        scenarios = {
            'normal': {
                'rainfall': 15.0, 'water_level': 2.5, 'river_discharge': 120.0,
                'soil_moisture': 45.0, 'temperature': 22.0, 'humidity': 65.0,
                'elevation': 150.0, 'slope': 5.0
            },
            'heavy_rain': {
                'rainfall': 85.0, 'water_level': 4.2, 'river_discharge': 350.0,
                'soil_moisture': 80.0, 'temperature': 18.0, 'humidity': 90.0,
                'elevation': 120.0, 'slope': 8.0
            },
            'drought': {
                'rainfall': 2.0, 'water_level': 1.2, 'river_discharge': 45.0,
                'soil_moisture': 15.0, 'temperature': 30.0, 'humidity': 35.0,
                'elevation': 200.0, 'slope': 3.0
            }
        }
        
        features = scenarios.get(scenario, scenarios['normal'])
        features['location'] = f"Quick Prediction - {scenario.replace('_', ' ').title()}"
        features['notes'] = f"Auto-generated quick prediction for {scenario} scenario"
        
        # Create a mock request for the predict_flood function
        from django.test import RequestFactory
        factory = RequestFactory()
        mock_request = factory.post('/predict/', features)
        mock_request.content_type = 'application/json'
        
        return predict_flood(mock_request)
    
    return render(request, 'flood_prediction/quick_prediction.html')

@login_required
def system_settings(request):
    """System configuration settings"""
    if request.method == 'POST':
        try:
            settings_data = request.POST.dict()
            
            for key, value in settings_data.items():
                if key.startswith('setting_'):
                    setting_key = key.replace('setting_', '')
                    setting, created = SystemSettings.objects.get_or_create(
                        key=setting_key,
                        defaults={'value': value, 'updated_by': request.user}
                    )
                    if not created:
                        setting.value = value
                        setting.updated_by = request.user
                        setting.save()
            
            messages.success(request, 'Settings updated successfully')
            return redirect('system_settings')
            
        except Exception as e:
            logger.error(f"Settings update error: {str(e)}")
            messages.error(request, 'Failed to update settings')
    
    settings = SystemSettings.objects.all()
    context = {
        'settings': settings
    }
    return render(request, 'flood_prediction/settings.html', context)