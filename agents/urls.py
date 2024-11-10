from django.urls import path
from . import views

from .views import (
    AssignmentProcessingView,
    AssignmentReviewView,
    AssignmentStatusView
)

app_name = 'agents'

urlpatterns = [
    path('agent-assignments/', views.AgentAssignmentListView.as_view(), name='agent-assignments'),
    path('child-images/', views.ChildImageListView.as_view(), name='child-images'),
    path('child-user-containers/', views.ChildUserContainerListView.as_view(), name='child-user-containers'),
    path(
        'assignments/<int:assignment_id>/process/',
        AssignmentProcessingView.as_view(),
        name='assignment-process'
    ),
    path(
        'assignments/<int:assignment_id>/review/',
        AssignmentReviewView.as_view(),
        name='assignment-review'
    ),
    path(
        'assignments/<int:assignment_id>/status/',
        AssignmentStatusView.as_view(),
        name='assignment-status'
    ),
]