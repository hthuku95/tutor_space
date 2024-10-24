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
    agent = serializers.StringRelatedField()
    original_platform = OriginalPlatformSerializer()
    original_account = FreelancingAccountSerializer()
    assignment_files = AssignmentFileSerializer(many=True)
    assignment_submissions = AssignmentSubmissionSerializer(many=True)
    revisions = RevisionSerializer(many=True)

    class Meta:
        model = Assignment
        fields = '__all__'


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


