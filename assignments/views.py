from rest_framework import generics, permissions
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
from .serializers import (
    OriginalPlatformSerializer,
    FreelancingAccountSerializer,
    FileSerializer,
    AssignmentSerializer,
    ChatSerializer,
    AttachmentSerializer,
    MessageSerializer,
    AssignmentFileSerializer,
    AssignmentSubmissionSerializer,
    RevisionFileSerializer,
    RevisionSerializer,
    SearchTagPairsSerializer,
)
from django.shortcuts import get_object_or_404
from rest_framework.permissions import IsAdminUser
from agents.main_agent_one import run_bidding_process
from rest_framework.views import APIView
from rest_framework.response import Response

class TriggerBiddingView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request):
        try:
            run_bidding_process()
            return Response({"message": "Bidding process completed successfully"}, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=500)


class OriginalPlatformListView(generics.ListAPIView):
    queryset = OriginalPlatform.objects.all()
    serializer_class = OriginalPlatformSerializer
    permission_classes = [permissions.IsAuthenticated]


class OriginalPlatformDetailView(generics.RetrieveAPIView):
    serializer_class = OriginalPlatformSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        pk = self.kwargs.get('pk')
        return get_object_or_404(OriginalPlatform, pk=pk)


class FreelancingAccountListView(generics.ListAPIView):
    queryset = FreelancingAccount.objects.all()
    serializer_class = FreelancingAccountSerializer
    permission_classes = [permissions.IsAuthenticated]


class FreelancingAccountDetailView(generics.RetrieveAPIView):
    serializer_class = FreelancingAccountSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        pk = self.kwargs.get('pk')
        return get_object_or_404(FreelancingAccount, pk=pk)


class FileListView(generics.ListAPIView):
    queryset = File.objects.all()
    serializer_class = FileSerializer
    permission_classes = [permissions.IsAuthenticated]


class FileDetailView(generics.RetrieveAPIView):
    serializer_class = FileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        pk = self.kwargs.get('pk')
        return get_object_or_404(File, pk=pk)


class AssignmentListView(generics.ListAPIView):
    queryset = Assignment.objects.all()
    serializer_class = AssignmentSerializer
    permission_classes = [permissions.IsAuthenticated]


class AssignmentDetailView(generics.RetrieveAPIView):
    serializer_class = AssignmentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        pk = self.kwargs.get('pk')
        return get_object_or_404(Assignment, pk=pk)


class ChatListView(generics.ListAPIView):
    queryset = Chat.objects.all()
    serializer_class = ChatSerializer
    permission_classes = [permissions.IsAuthenticated]


class ChatDetailView(generics.RetrieveAPIView):
    serializer_class = ChatSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        pk = self.kwargs.get('pk')
        return get_object_or_404(Chat, pk=pk)


class AttachmentListView(generics.ListAPIView):
    queryset = Attachment.objects.all()
    serializer_class = AttachmentSerializer
    permission_classes = [permissions.IsAuthenticated]


class AttachmentDetailView(generics.RetrieveAPIView):
    serializer_class = AttachmentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        pk = self.kwargs.get('pk')
        return get_object_or_404(Attachment, pk=pk)


class MessageListView(generics.ListAPIView):
    queryset = Message.objects.all()
    serializer_class = MessageSerializer
    permission_classes = [permissions.IsAuthenticated]


class MessageDetailView(generics.RetrieveAPIView):
    serializer_class = MessageSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        pk = self.kwargs.get('pk')
        return get_object_or_404(Message, pk=pk)


class AssignmentFileListView(generics.ListAPIView):
    queryset = AssignmentFile.objects.all()
    serializer_class = AssignmentFileSerializer
    permission_classes = [permissions.IsAuthenticated]


class AssignmentFileDetailView(generics.RetrieveAPIView):
    serializer_class = AssignmentFileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        pk = self.kwargs.get('pk')
        return get_object_or_404(AssignmentFile, pk=pk)


class AssignmentSubmissionListView(generics.ListAPIView):
    queryset = AssignmentSubmission.objects.all()
    serializer_class = AssignmentSubmissionSerializer
    permission_classes = [permissions.IsAuthenticated]


class AssignmentSubmissionDetailView(generics.RetrieveAPIView):
    serializer_class = AssignmentSubmissionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        pk = self.kwargs.get('pk')
        return get_object_or_404(AssignmentSubmission, pk=pk)


class RevisionFileListView(generics.ListAPIView):
    queryset = RevisionFile.objects.all()
    serializer_class = RevisionFileSerializer
    permission_classes = [permissions.IsAuthenticated]


class RevisionFileDetailView(generics.RetrieveAPIView):
    serializer_class = RevisionFileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        pk = self.kwargs.get('pk')
        return get_object_or_404(RevisionFile, pk=pk)


class RevisionListView(generics.ListAPIView):
    queryset = Revision.objects.all()
    serializer_class = RevisionSerializer
    permission_classes = [permissions.IsAuthenticated]


class RevisionDetailView(generics.RetrieveAPIView):
    serializer_class = RevisionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        pk = self.kwargs.get('pk')
        return get_object_or_404(Revision, pk=pk)
    
class SearchTagPairsListView(generics.ListAPIView):
    queryset = SearchTagPairs.objects.all()
    serializer_class = SearchTagPairsSerializer
    permission_classes = [permissions.IsAuthenticated]
