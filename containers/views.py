from django.shortcuts import render,redirect
from .models import UserContainer, Image
from django.contrib import messages
from django.contrib.auth.decorators import login_required

# Create your views here.
@login_required()
def containers(request):
    containers = UserContainer.objects.all()
    images = Image.objects.all()

    context = {
        'containers':containers,
        'images':images
    }

    return render(request, "containers/containers.html",context)

