from rest_framework import serializers
from .models import AgentAssignment, ChildImage, ChildUserContainer
from assignments.models import Assignment, AssignmentSubmission

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

class AssignmentSubmissionSerializer(serializers.ModelSerializer):
    """Serializer for assignment submission data"""
    class Meta:
        model = AssignmentSubmission
        fields = [
            'id',
            'date_completed',
            'date_to_be_delivered',
            'version',
            'status'
        ]

class AssignmentStatusSerializer(serializers.ModelSerializer):
    """Serializer for assignment status data"""
    latest_submission = AssignmentSubmissionSerializer(read_only=True)
    assignment_type_display = serializers.CharField(source='get_assignment_type_display')

    class Meta:
        model = Assignment
        fields = [
            'id',
            'subject',
            'description',
            'assignment_type',
            'assignment_type_display',
            'completed',
            'has_deposit_been_paid',
            'has_revisions',
            'completion_deadline',
            'expected_delivery_time',
            'latest_submission'
        ]

class ReviewResultSerializer(serializers.Serializer):
    """Serializer for review results"""
    passed = serializers.BooleanField()
    issues = serializers.ListField(
        child=serializers.DictField(),
        required=False
    )
    suggestions = serializers.ListField(
        child=serializers.DictField(),
        required=False
    )
    details = serializers.DictField(required=False)

class ProcessingResultSerializer(serializers.Serializer):
    """Serializer for processing results"""
    status = serializers.CharField()
    assignment_id = serializers.IntegerField()
    execution_plan = serializers.DictField(required=False)
    result = serializers.DictField()