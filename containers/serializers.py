# containers/serializers.py
from rest_framework import serializers
from .models import UserContainer, Image, Project

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = '__all__' 


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = '__all__'


class UserContainerSerializer(serializers.ModelSerializer):
    # Ensure this field is used to reference existing images only
    image = serializers.PrimaryKeyRelatedField(queryset=Image.objects.all())

    class Meta:
        model = UserContainer
        fields = ('container_id', 'container_name', 'user_profile', 'image')

    def create(self, validated_data):
        # Here, validated_data['image'] will be an Image instance (automatically fetched by DRF)
        container = UserContainer.objects.create(**validated_data)
        return container

    def update(self, instance, validated_data):
        # Here, validated_data['image'] will also be an Image instance if provided
        instance.container_name = validated_data.get('container_name', instance.container_name)
        instance.image = validated_data.get('image', instance.image)  # Handle change of image if necessary
        instance.save()
        return instance
