from rest_framework import serializers
from .models import (
    OriginalPlatform,
    FreelancingAccount,
    File,
    Assignment,
    Chat,
    Attachment,
    Message,
    AssignmentFile,
    AssignmentSubmission,
    RevisionFile,
    Revision,
    SearchTagPairs,
)
from profiles.serializers import UserProfileSerializer

class OriginalPlatformSerializer(serializers.ModelSerializer):
    class Meta:
        model = OriginalPlatform
        fields = '__all__'


class FreelancingAccountSerializer(serializers.ModelSerializer):
    original_platform = OriginalPlatformSerializer()

    class Meta:
        model = FreelancingAccount
        fields = '__all__'


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = File
        fields = '__all__'


class AttachmentSerializer(serializers.ModelSerializer):
    files = FileSerializer(many=True)

    class Meta:
        model = Attachment
        fields = '__all__'


class AssignmentFileSerializer(serializers.ModelSerializer):
    files = FileSerializer(many=True)

    class Meta:
        model = AssignmentFile
        fields = '__all__'


class AssignmentSubmissionSerializer(serializers.ModelSerializer):
    files = FileSerializer(many=True)

    class Meta:
        model = AssignmentSubmission
        fields = '__all__'


class RevisionFileSerializer(serializers.ModelSerializer):
    files = FileSerializer(many=True)

    class Meta:
        model = RevisionFile
        fields = '__all__'


class RevisionSerializer(serializers.ModelSerializer):
    assignment_submission = AssignmentSubmissionSerializer()
    revision_files = RevisionFileSerializer(many=True)

    class Meta:
        model = Revision
        fields = '__all__'


class AssignmentSerializer(serializers.ModelSerializer):
    agent = UserProfileSerializer(read_only=True)
    delivery_status = serializers.DictField(read_only=True)
    can_deliver = serializers.BooleanField(read_only=True)
    expected_delivery_time = serializers.DateTimeField(read_only=True)

    def get_can_access_submission(self, obj):
        request = self.context.get('request')
        if request and request.user:
            return obj.can_access_submission(request.user)
        return False

    class Meta:
        model = Assignment
        fields = [
            'id',
            'agent',
            'subject',
            'description',
            'assignment_type',
            'completed',
            'has_revisions',
            'has_deposit_been_paid',
            'completion_deadline',
            'expected_delivery_time',
            'delivery_status',
            'can_deliver',
            'can_access_submission'
        ]

class MessageSerializer(serializers.ModelSerializer):
    attachments = AttachmentSerializer()

    class Meta:
        model = Message
        fields = '__all__'


class ChatSerializer(serializers.ModelSerializer):
    assignment = AssignmentSerializer()
    messages = MessageSerializer(many=True)

    class Meta:
        model = Chat
        fields = '__all__'


class SearchTagPairsSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = SearchTagPairs
        fields = '__all__'


