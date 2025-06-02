from django.db import models


# Create your models here.
class TitanicUserRegistrationModel(models.Model):
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
        db_table = 'TitanicUsers'


class TrainingModel(models.Model):
    PassengerId = models.IntegerField(default=0,null=True)
    Survived = models.IntegerField(default=0,null=True)
    Pclass = models.IntegerField(default=0,null=True)
    Name = models.CharField(max_length=1000,null=True)
    Sex = models.CharField(max_length=100,null=True)
    Age = models.FloatField(default=0,null=True)
    SibSp = models.IntegerField(default=0,null=True)
    Parch = models.IntegerField(default=0,null=True)
    Ticket = models.CharField(max_length=100,null=True)
    Fare = models.FloatField(default=0,null=True)
    Cabin = models.CharField(max_length=100,null=True)
    Embarked = models.CharField(max_length=100,null=True)

    def __str__(self):
        return self.PassengerId
    class Meta:
        db_table = "TrainingDataset"

class TestingModel(models.Model):
    PassengerId = models.IntegerField(default=0,null=True)
    Pclass = models.IntegerField(default=0,null=True)
    Name = models.CharField(max_length=500,null=True)
    Sex = models.CharField(max_length=100,null=True)
    Age = models.FloatField(default=0,null=True)
    SibSp = models.IntegerField(default=0,null=True)
    Parch = models.IntegerField(default=0,null=True)
    Ticket = models.CharField(max_length=100,null=True)
    Fare = models.FloatField(default=0,null=True)
    Cabin = models.CharField(max_length=100,null=True)
    Embarked = models.CharField(max_length=100,null=True)

    def __str__(self):
        return self.PassengerId
    class Meta:
        db_table = "TestingDataset"
