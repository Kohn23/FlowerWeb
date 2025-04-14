# Create your models here.
# models.py
from django.db import models


class FlowerDetail(models.Model):
    flower_type = models.CharField(max_length=100, unique=True)
    sunlight = models.CharField(max_length=50)
    water = models.CharField(max_length=50)
    temperature = models.CharField(max_length=50)
    season = models.CharField(max_length=50)
    soil = models.CharField(max_length=100)
    height = models.CharField(max_length=50)
    spread = models.CharField(max_length=50)
    lifespan = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

