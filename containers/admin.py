from django.contrib import admin
from .models import UserContainer,Image, Project
# Register your models here.
admin.site.register(UserContainer)
admin.site.register(Image)
admin.site.register(Project)