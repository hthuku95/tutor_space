from rest_framework.generics import ListAPIView
from rest_framework.permissions import IsAuthenticated
from .models import (
    AgentAssignment, ChildImage, ChildUserContainer
)
from rest_framework.views import APIView
from .serializers import (
    AgentAssignmentSerializer, ChildImageSerializer, ChildUserContainerSerializer
)
from rest_framework.response import Response
from rest_framework import status
import logging

logger = logging.getLogger(__name__)

class AgentAssignmentListView(ListAPIView):
    queryset = AgentAssignment.objects.all()
    serializer_class = AgentAssignmentSerializer
    permission_classes = [IsAuthenticated]

class ChildImageListView(ListAPIView):
    queryset = ChildImage.objects.all()
    serializer_class = ChildImageSerializer
    permission_classes = [IsAuthenticated]

class ChildUserContainerListView(ListAPIView):
    queryset = ChildUserContainer.objects.all()
    serializer_class = ChildUserContainerSerializer
    permission_classes = [IsAuthenticated]

