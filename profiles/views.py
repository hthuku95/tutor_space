from django.shortcuts import render,redirect
from .models import UserProfile
from django.contrib import messages
from django.contrib.auth.decorators import login_required

# Create your views here.
@login_required()
def dashboard(request):
    user_profile = UserProfile.objects.get(user=request.user)

    context = {
        'user_profile':user_profile
    }

    return render(request, "profiles/dashboard.html",context)

