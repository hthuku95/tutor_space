from django.db import models
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from assignments.models import Assignment,FreelancingAccount
from profiles.models import UserProfile
from containers.models import Image, UserContainer

class Agent(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    is_active = models.BooleanField(default=True)

    class Meta:
        abstract = True

class AgentAssignment(models.Model):
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    agent = GenericForeignKey('content_type', 'object_id')
    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE)
    assigned_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('content_type', 'object_id', 'assignment')

class ChildImage(models.Model):
    base_image = models.ForeignKey(Image, on_delete=models.CASCADE)
    dockerfile_content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

class ChildUserContainer(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    child_image = models.ForeignKey(ChildImage, on_delete=models.CASCADE)
    container_id = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)