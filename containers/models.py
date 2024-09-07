from django.db import models
from profiles.models import UserProfile


class Image(models.Model):
    image_id = models.CharField(max_length=100,blank=False,null=False)
    repository = models.CharField(max_length=100)
    tag = models.CharField(max_length=100)
    size = models.CharField(max_length=100)
    created = models.DateTimeField(auto_now=False,auto_now_add=False)

    def __str__(self):
        return self.image_id


class UserContainer(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    container_id = models.CharField(max_length=255)
    container_name = models.CharField(max_length=255)
    image = models.ForeignKey(Image,on_delete=models.CASCADE)

    def __str__(self):
        return self.container_id


