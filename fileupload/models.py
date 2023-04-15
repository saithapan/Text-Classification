from django.db import models


# Create your models here.

class FileData(models.Model):
    text = models.TextField()
    number = models.TextField()
