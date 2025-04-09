from django.db import models
# Create your models here.
class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(unique=True, max_length=100)
    email = models.CharField(unique=True, max_length=100)
    locality = models.CharField(max_length=100)
    address = models.CharField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'UserRegistrations'



class UserAppointmentModel(models.Model):
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    phone = models.CharField(max_length=100)
    date = models.CharField(max_length=100)
    doctor = models.CharField(max_length=100)
    disease = models.CharField(max_length=100)
    suggestion = models.TextField(max_length=1000)
    status = models.CharField(max_length=100, default='waiting')


    def __str__(self):
        return self.name

    class Meta:
        db_table = 'appointmenttable'
