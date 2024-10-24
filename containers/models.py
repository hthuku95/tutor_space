from django.db import models
from profiles.models import UserProfile



class Project(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    project_name = models.CharField(max_length=255)
    project_type = models.CharField(max_length=50)  # e.g., 'react', 'python'
    created_at = models.DateTimeField(auto_now_add=True)
    directory = models.CharField(max_length=500)  # Path to the project directory

    def __str__(self):
        return f"{self.project_name} ({self.project_type})"
       
class Image(models.Model):
    image_id = models.CharField(max_length=100, blank=False, null=False, primary_key=True)
    repository = models.CharField(max_length=100)
    tag = models.CharField(max_length=100)
    language_for_execution = models.CharField(blank=True, null=True, max_length=100)
    execution_flags = models.CharField(blank=True,null=True,max_length=20)
    created = models.DateTimeField(auto_now_add=True)
    default_command = models.CharField(blank=True, null=True, max_length=100)

    def __str__(self):
        return self.image_id


class UserContainer(models.Model):
    container_id = models.CharField(max_length=255,primary_key=True)
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    container_name = models.CharField(max_length=255)
    image = models.ForeignKey(Image,on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return self.container_id
    



