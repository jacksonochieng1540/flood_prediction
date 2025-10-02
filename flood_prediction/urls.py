# flood_prediction/urls.py

from django.urls import path
from . import views

app_name = 'flood_prediction'

urlpatterns = [
    path('', views.index, name='index'),
    path('history/', views.prediction_history, name='prediction_history'),
    path('prediction/<int:prediction_id>/', views.prediction_detail, name='prediction_detail'),
    path('alerts/', views.alerts_dashboard, name='alerts_dashboard'),
    path('analytics/', views.analytics_dashboard, name='analytics_dashboard'),
    path('importance/', views.feature_importance, name='feature_importance'),
    path('quick-predict/', views.quick_prediction, name='quick_prediction'),
    path('settings/', views.system_settings, name='system_settings'),
    
    # Authentication pages
    path('accounts/login/', views.login_view, name='login'),
    path('accounts/logout/', views.logout_view, name='logout'),
    path('accounts/register/', views.register_view, name='register'),
    path('accounts/profile/', views.profile_view, name='profile'),
    
    # API endpoints
    path('api/predict/', views.predict_flood, name='predict_flood'),
    path('api/predict/batch/', views.batch_predict, name='batch_predict'),
    path('api/alerts/<int:alert_id>/update/', views.update_alert_status, name='update_alert_status'),
    path('api/model/retrain/', views.retrain_model, name='retrain_model'),
    path('api/prediction/<int:prediction_id>/verify/', views.verify_prediction_outcome, name='verify_prediction_outcome'),
    
    # Export and utilities
    path('export/csv/', views.export_predictions_csv, name='export_predictions_csv'),
    path('health/', views.health_check, name='health_check'),
]

# Error handlers
handler404 = views.handler404
handler500 = views.handler500