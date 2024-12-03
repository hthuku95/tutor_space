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

from django.db.models import Q
import logging
from rest_framework import status
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
from rest_framework.permissions import IsAuthenticated

logger = logging.getLogger(__name__)

class TriggerBiddingView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request):
        try:
            run_bidding_process()
            return Response({"message": "Bidding process completed successfully"}, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
        

class MessageImprovementView(APIView):
    permission_classes = [IsAuthenticated]
    
    async def post(self, request, assignment_id):
        """
        Handle message improvement through Agent 2
        Only improves messages from us (creator=True) before sending to client
        """
        try:
            # Get assignment and validate access
            assignment = await Assignment.objects.aget(pk=assignment_id)
            if request.user != assignment.agent.user:
                return Response(
                    {"error": "Not authorized"},
                    status=status.HTTP_403_FORBIDDEN
                )

            # Get message content
            message_content = request.data.get('message')
            if not message_content:
                return Response(
                    {"error": "No message provided"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get or create chat
            chat = await Chat.objects.aget_or_create(
                assignment=assignment
            )[0]

            # Store original message first (not sent to client yet)
            original_message = await Message.objects.acreate(
                chat=chat,
                body=message_content,
                creator=True,  # This is our message
            )

            # Get improved version from Agent 2
            improved_message = await self._get_message_improvement(
                message_content,
                assignment,
                chat
            )

            return Response({
                "message_id": original_message.id,
                "original": message_content,
                "improved": improved_message,
                "can_deliver": assignment.can_deliver,
                "delivery_status": assignment.delivery_status
            })

        except Assignment.DoesNotExist:
            return Response(
                {"error": "Assignment not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error in message improvement: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def patch(self, request, assignment_id, message_id):
        """
        Accept or reject improved message and mark as final
        """
        try:
            # Get message and validate
            message = await Message.objects.aget(
                id=message_id,
                chat__assignment_id=assignment_id,
                creator=True  # Must be our message
            )

            # Validate user access
            if request.user != message.chat.assignment.agent.user:
                return Response(
                    {"error": "Not authorized"},
                    status=status.HTTP_403_FORBIDDEN
                )

            # Get the final version chosen by human agent
            final_version = request.data.get('final_version')
            if not final_version:
                return Response(
                    {"error": "No final version provided"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Update the message with final version
            message.body = final_version
            await message.asave()

            return Response({
                "message_id": message.id,
                "final_version": final_version
            })

        except Message.DoesNotExist:
            return Response(
                {"error": "Message not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error updating message: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def get(self, request, assignment_id):
        """Get chat messages for an assignment"""
        try:
            # Get assignment and validate access
            assignment = await Assignment.objects.aget(pk=assignment_id)
            if request.user != assignment.agent.user:
                return Response(
                    {"error": "Not authorized"},
                    status=status.HTTP_403_FORBIDDEN
                )

            # Get chat and messages
            chat = await Chat.objects.filter(
                assignment=assignment
            ).prefetch_related('messages').afirst()

            if not chat:
                return Response([])

            # Get messages with attachments
            messages = await self._get_messages(chat)
            
            return Response({
                "assignment": AssignmentSerializer(assignment).data,
                "messages": messages
            })

        except Assignment.DoesNotExist:
            return Response(
                {"error": "Assignment not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error getting messages: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def _get_messages(self, chat: Chat) -> List[Dict]:
        """Get formatted messages for a chat"""
        try:
            messages = []
            async for message in chat.messages.all():
                msg_data = {
                    "id": message.id,
                    "body": message.body,
                    "creator": "Us" if message.creator else "Client",
                    "timestamp": message.timestamp.isoformat()
                }
                
                # Add attachments if any
                attachments = await message.attachments.all()
                if attachments:
                    msg_data["attachments"] = [
                        {
                            "id": att.id,
                            "files": [
                                {
                                    "id": f.id,
                                    "url": f.file.url,
                                    "name": f.file.name
                                } for f in await att.files.all()
                            ]
                        } for att in attachments
                    ]

                messages.append(msg_data)

            return messages

        except Exception as e:
            logger.error(f"Error formatting messages: {str(e)}")
            raise

    async def _get_message_improvement(
        self,
        message: str,
        assignment: Assignment,
        chat: Chat
    ) -> str:
        """Get message improvement from Agent 2"""
        try:
            # Get recent messages for context
            recent_messages = await Message.objects.filter(
                chat=chat
            ).order_by('-timestamp')[:5].values('body', 'creator', 'timestamp')

            # Format message history for context
            message_history = [
                f"{'Us' if msg['creator'] else 'Client'}: {msg['body']}"
                for msg in recent_messages
            ]

            # Initialize LLM
            model = ChatOpenAI(model="gpt-4")
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a professional freelancer assistant helping improve message quality.
                Assignment Type: {assignment_type}
                Delivery Status: {delivery_status}
                
                Recent conversation:
                {message_history}
                
                Improve the message while:
                1. Maintaining professional tone
                2. Ensuring clarity and completeness
                3. Following freelancing best practices
                4. Considering delivery status and timeline
                5. Preserving essential information
                
                If the message mentions delivery and it's not yet allowed,
                suggest appropriate alternative responses."""),
                ("human", "{message}")
            ])

            # Get improvement
            result = await prompt.ainvoke({
                "assignment_type": assignment.get_assignment_type_display(),
                "delivery_status": assignment.delivery_status,
                "message_history": "\n".join(message_history),
                "message": message
            })

            return result.content

        except Exception as e:
            logger.error(f"Error getting message improvement: {str(e)}")
            raise


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
    serializer_class = AssignmentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Get base queryset - only assignments for the logged-in user's profile
        queryset = Assignment.objects.filter(agent__user=self.request.user)

        # Get query parameters
        filter_param = self.request.query_params.get('filter', 'all')
        search_query = self.request.query_params.get('search', '')
        sort_by = self.request.query_params.get('sort', 'deadline')

        # Apply filters
        if filter_param == 'completed':
            queryset = queryset.filter(completed=True)
        elif filter_param == 'in_progress':
            queryset = queryset.filter(completed=False)
        elif filter_param == 'revisions':
            queryset = queryset.filter(has_revisions=True)
        elif filter_param == 'unpaid':
            queryset = queryset.filter(has_deposit_been_paid=False)

        # Apply search if provided
        if search_query:
            queryset = queryset.filter(
                Q(subject__icontains=search_query) |
                Q(description__icontains=search_query)
            )

        # Apply sorting
        if sort_by == 'deadline':
            queryset = queryset.order_by('completion_deadline')
        elif sort_by == 'created':
            queryset = queryset.order_by('-timestamp')
        elif sort_by == 'subject':
            queryset = queryset.order_by('subject')

        return queryset

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
