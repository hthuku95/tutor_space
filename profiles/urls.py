from django.urls import path
from . import views

app_name = 'profiles'

urlpatterns = [
    path(r'',views.dashboard,name='dashboard'),
]