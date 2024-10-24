
from django.contrib import admin
from django.urls import path,include
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static
from django.conf import settings
from . import views

from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),
    path(r'',views.index_view),
    path(r'api/dashboard/',include('profiles.urls')),
    path(r'api/containers/',include('containers.urls')),
    path(r'api/assignments/',include('assignments.urls')),
    path(r'api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path(r'api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path(r'api/auth/', include('dj_rest_auth.urls')),
    path(r'api/auth/registration/', include('dj_rest_auth.registration.urls')),
    path(r'api/agents/', include('agents.urls')),
]

#appending the static files urls to the above media
urlpatterns += staticfiles_urlpatterns()
#how to upload media..appending the media url to the patterns above