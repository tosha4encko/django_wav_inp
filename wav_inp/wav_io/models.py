from django.db import models

class WavIO(models.Model):
    name = models.CharField(max_length=100)
    albom = models.CharField(max_length=100)
    artist = models.CharField(max_length=100)

    data = models.CharField(max_length=3000, blank=True)
    ftag = models.CharField(max_length=3000, blank=True)

    clust = models.IntegerField(blank=True)
    dist = models.IntegerField(blank=True)
    yandex_au_url = models.CharField(max_length=3000, blank=True)
