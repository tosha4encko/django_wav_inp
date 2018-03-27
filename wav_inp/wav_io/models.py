from django.db import models

class WavIO(models.Model):
    name = models.CharField(max_length=100)
    tag = models.IntegerField()
    data = models.CharField(max_length=300)
    dist = models.IntegerField()

