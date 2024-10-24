from django.urls import path
from . import views

app_name = 'agents'

urlpatterns = [
    path('agent-assignments/', views.AgentAssignmentListView.as_view(), name='agent-assignments'),
    path('child-images/', views.ChildImageListView.as_view(), name='child-images'),
    path('child-user-containers/', views.ChildUserContainerListView.as_view(), name='child-user-containers'),
]