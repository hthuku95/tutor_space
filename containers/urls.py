from django.urls import path
from .views import ContainerListView, ImageView,ExecuteCodeView,GenerateCodeView,GenerateApplicationView

app_name = 'containers'

urlpatterns = [
    path('', ContainerListView.as_view(), name='containers'),
    path('images', ImageView.as_view(), name='images'),
    path('execute', ExecuteCodeView.as_view(), name='execute_code'), 
    path('generate', GenerateCodeView.as_view(), name='generate_code'),
    path('generate-application', GenerateApplicationView.as_view(), name='generate_application'),
]