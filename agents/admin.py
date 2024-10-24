from django.contrib import admin
from .models import (
    AgentAssignment,
    ChildImage,
    ChildUserContainer
)

@admin.register(AgentAssignment)
class AgentAssignmentAdmin(admin.ModelAdmin):
    list_display = ('agent', 'assignment', 'assigned_at', 'completed_at')
    list_filter = ('assigned_at', 'completed_at')
    search_fields = ('agent__name', 'assignment__subject')

@admin.register(ChildImage)
class ChildImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'base_image', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('base_image__image_id',)

@admin.register(ChildUserContainer)
class ChildUserContainerAdmin(admin.ModelAdmin):
    list_display = ('id', 'user_profile', 'child_image', 'container_id', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user_profile__user__username', 'container_id')