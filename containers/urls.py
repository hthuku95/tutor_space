from django.urls import path
from .views import ContainerListView, ImageView,ExecuteCodeView

app_name = 'containers'

urlpatterns = [
    path('', ContainerListView.as_view(), name='containers'),
    path('images', ImageView.as_view(), name='images'),
    path('execute', ExecuteCodeView.as_view(), name='execute_code'),  # Make sure ExecuteCode is also using DRF
]