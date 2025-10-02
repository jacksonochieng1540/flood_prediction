
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone


class Location(models.Model):
    name = models.CharField(max_length=100, unique=True)
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    elevation = models.FloatField(help_text="Elevation in meters")
    river_basin = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.latitude}, {self.longitude})"


class WeatherData(models.Model):
    """Historical and current weather data"""
    location = models.ForeignKey(Location, on_delete=models.CASCADE, related_name='weather_data')
    timestamp = models.DateTimeField()
    rainfall_mm = models.FloatField(validators=[MinValueValidator(0)], help_text="Rainfall in mm")
    temperature_celsius = models.FloatField(help_text="Temperature in Celsius")
    humidity_percent = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Humidity percentage"
    )
    wind_speed_kmh = models.FloatField(validators=[MinValueValidator(0)], help_text="Wind speed in km/h")
    pressure_hpa = models.FloatField(help_text="Atmospheric pressure in hPa")
    
    class Meta:
        ordering = ['-timestamp']
        unique_together = ['location', 'timestamp']
    
    def __str__(self):
        return f"Weather at {self.location.name} - {self.timestamp}"


class PredictionHistory(models.Model):
    """Store historical predictions and their outcomes"""
    RISK_LEVEL_CHOICES = [
        ('very_low', 'Very Low'),
        ('low', 'Low'),
        ('moderate', 'Moderate'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    # Input features
    rainfall_mm = models.FloatField(validators=[MinValueValidator(0)])
    water_level_m = models.FloatField(validators=[MinValueValidator(0)])
    river_discharge_m3s = models.FloatField(validators=[MinValueValidator(0)])
    soil_moisture_percent = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    temperature_celsius = models.FloatField()
    humidity_percent = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    elevation_m = models.FloatField()
    slope_degree = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(90)])
    
    # Prediction results
    prediction = models.BooleanField(help_text="True if flood predicted, False otherwise")
    probability_flood = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Probability of flood occurrence (0-1)"
    )
    probability_no_flood = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Probability of no flood (0-1)"
    )
    risk_level = models.CharField(max_length=20, choices=RISK_LEVEL_CHOICES)
    confidence = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        help_text="Model confidence in prediction"
    )
    model_used = models.CharField(max_length=50, default='ensemble')
    
    # Metadata
    location = models.CharField(max_length=100, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Actual outcome (to be filled later for model improvement)
    actual_flood_occurred = models.BooleanField(null=True, blank=True)
    actual_flood_severity = models.CharField(max_length=20, blank=True)
    verified_at = models.DateTimeField(null=True, blank=True)
    verified_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['created_at']),
            models.Index(fields=['risk_level']),
            models.Index(fields=['location']),
        ]
    
    def __str__(self):
        return f"Prediction {self.id} - {self.risk_level} risk - {self.created_at}"
    
    @property
    def is_high_risk(self):
        return self.risk_level in ['high', 'critical']
    
    @property
    def flood_probability_percent(self):
        return round(self.probability_flood * 100, 1)
    
    @property
    def confidence_percent(self):
        return round(self.confidence * 100, 1)


class Alert(models.Model):
    """Flood alerts and warnings"""
    ALERT_TYPES = [
        ('flood_warning', 'Flood Warning'),
        ('heavy_rain', 'Heavy Rain Alert'),
        ('river_level', 'High River Level'),
        ('soil_saturation', 'Soil Saturation Alert'),
    ]
    
    SEVERITY_LEVELS = [
        ('low', 'Low'),
        ('moderate', 'Moderate'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    ALERT_STATUS = [
        ('active', 'Active'),
        ('resolved', 'Resolved'),
        ('cancelled', 'Cancelled'),
    ]
    
    prediction = models.ForeignKey(
        PredictionHistory, 
        on_delete=models.CASCADE, 
        related_name='alerts'
    )
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS)
    message = models.TextField()
    affected_area = models.CharField(max_length=200)
    recommended_actions = models.TextField(blank=True)
    
    # Alert management
    status = models.CharField(max_length=20, choices=ALERT_STATUS, default='active')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolved_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='resolved_alerts'
    )
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['severity']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"{self.get_alert_type_display()} - {self.severity} - {self.affected_area}"
    
    @property
    def duration(self):
        if self.resolved_at:
            return self.resolved_at - self.created_at
        return timezone.now() - self.created_at
    
    def resolve(self, user=None):
        self.status = 'resolved'
        self.resolved_at = timezone.now()
        if user:
            self.resolved_by = user
        self.save()


class ModelPerformance(models.Model):
    """Track model performance over time"""
    accuracy = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    precision = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    recall = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    f1_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    training_samples = models.IntegerField()
    model_version = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Model performances"
    
    def __str__(self):
        return f"Model v{self.model_version} - Accuracy: {self.accuracy:.3f} - {self.created_at.date()}"


class SystemSettings(models.Model):
    """System configuration settings"""
    key = models.CharField(max_length=100, unique=True)
    value = models.TextField()
    description = models.TextField(blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    
    def __str__(self):
        return self.key


# Signals for automated actions
from django.db.models.signals import post_save
from django.dispatch import receiver


@receiver(post_save, sender=PredictionHistory)
def create_high_risk_alert(sender, instance, created, **kwargs):
    """Automatically create alerts for high-risk predictions"""
    if created and instance.is_high_risk:
        Alert.objects.create(
            prediction=instance,
            alert_type='flood_warning',
            severity=instance.risk_level,
            message=f"High flood risk detected: {instance.get_risk_level_display()} risk "
                   f"with {instance.flood_probability_percent}% probability",
            affected_area=instance.location or "Unknown location",
            recommended_actions="Monitor water levels closely. Prepare evacuation plan if necessary."
        )