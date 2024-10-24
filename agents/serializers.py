from rest_framework import serializers
from .models import AgentAssignment, ChildImage, ChildUserContainer


class AgentAssignmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentAssignment
        fields = '__all__'

class ChildImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChildImage
        fields = '__all__'

class ChildUserContainerSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChildUserContainer
        fields = '__all__'