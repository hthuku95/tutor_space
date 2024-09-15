from rest_framework.generics import RetrieveAPIView
from .models import UserProfile
from .serializers import UserProfileSerializer
from django.contrib.auth.decorators import login_required
from rest_framework.permissions import IsAuthenticated

class DashboardView(RetrieveAPIView):
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user.userprofile
