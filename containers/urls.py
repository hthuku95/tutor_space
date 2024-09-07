from django.urls import path
from . import views

app_name = 'containers'

urlpatterns = [
    path(r'',views.containers,name='containers'),
]