from django.db import models

# Create your models here.
class Contact(models.Model):
	name=models.CharField(max_length=30)
	phone=models.CharField(max_length=12)
	email=models.CharField(max_length=30)
	msg=models.CharField(max_length=3000)