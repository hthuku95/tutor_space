from rest_framework.generics import ListAPIView
from rest_framework.permissions import IsAuthenticated
from .models import (
    AgentAssignment, ChildImage, ChildUserContainer
)
from django.http import JsonResponse
from rest_framework.views import APIView
from .serializers import (
    AgentAssignmentSerializer, ChildImageSerializer, ChildUserContainerSerializer
)
from .serializers import (
    AssignmentStatusSerializer,
    ReviewResultSerializer,
    ProcessingResultSerializer
)
from rest_framework.response import Response
from rest_framework import status
import logging
from typing import Dict, Any
from assignments.models import Assignment, AssignmentSubmission
from .main_agent_three import handle_assignment as agent_three_handle
from .main_agent_six import ReviewAgent


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


class AssignmentProcessingView(APIView):
    """
    View for handling assignment processing through Agent 3.
    """
    permission_classes = [IsAuthenticated]

    async def post(self, request, assignment_id: int) -> JsonResponse:
        """Start assignment processing"""
        try:
            # Validate assignment exists and is ready
            try:
                assignment = await Assignment.objects.aget(pk=assignment_id)
            except Assignment.DoesNotExist:
                return JsonResponse(
                    {"error": f"Assignment {assignment_id} not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Check if assignment is ready for processing
            if not assignment.has_deposit_been_paid:
                return JsonResponse(
                    {"error": "Assignment deposit not paid"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if assignment.completed:
                return JsonResponse(
                    {"error": "Assignment already completed"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            result = await agent_three_handle(assignment_id)
            
            serializer = ProcessingResultSerializer(data={
                "status": "success",
                "assignment_id": assignment_id,
                "result": result
            })
            if serializer.is_valid():
                return JsonResponse(serializer.data)
            return JsonResponse(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )

        except Exception as e:
            logger.error(f"Error processing assignment: {str(e)}")
            return JsonResponse(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AssignmentReviewView(APIView):
    """
    View for handling assignment review through Agent 6.
    """
    permission_classes = [IsAuthenticated]

    async def post(self, request, assignment_id: int) -> JsonResponse:
        """Start assignment review"""
        try:
            # Validate assignment exists
            try:
                assignment = await Assignment.objects.aget(pk=assignment_id)
            except Assignment.DoesNotExist:
                return JsonResponse(
                    {"error": f"Assignment {assignment_id} not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Get latest submission
            try:
                submission = await AssignmentSubmission.objects.filter(
                    assignment=assignment
                ).alatest('date_completed')
            except AssignmentSubmission.DoesNotExist:
                return JsonResponse(
                    {"error": "No submission found for review"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Initialize Agent 6 and review
            review_agent = ReviewAgent()
            review_result = await review_agent.review_assignment(
                assignment_id,
                submission.id
            )

            serializer = ReviewResultSerializer(data=review_result)
            if serializer.is_valid():
                return JsonResponse(serializer.data)
            return JsonResponse(
                serializer.errors, 
                status=status.HTTP_400_BAD_REQUEST
            )

        except Exception as e:
            logger.error(f"Error reviewing assignment: {str(e)}")
            return JsonResponse(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AssignmentStatusView(APIView):
    async def get(self, request, assignment_id: int) -> JsonResponse:
        try:
            assignment = await Assignment.objects.select_related(
                'agent', 
                'original_platform'
            ).prefetch_related(
                'submissions'
            ).aget(pk=assignment_id)
            
            serializer = AssignmentStatusSerializer(assignment)
            return JsonResponse(serializer.data)

        except Assignment.DoesNotExist:
            return JsonResponse(
                {"error": f"Assignment {assignment_id} not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error getting assignment status: {str(e)}")
            return JsonResponse(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )



'''

Before you continue with implementing the 60% rule, let me add some more context to it. this is what I want it to be:
1. remember AI is automating the process of doing the assignments. not humans so Ideally an App should be built within seconds or minutes at most. 
2. to continue on point 1, lets say a programming assignment is supposed to be finished in 1 month (30 days) and this application can complete it in minutes using AI, then the purpose of the 60% rule will be to calculate the delivery date, and in this case of 30 days, it will determine that the Assignment should be delivered in 18 days. so I am going to use number 3 to describe how I want it to be implemented
3. so in the frontend view I want the users to be able to see a list of assignments that have been assigned to them. What I mean is that all "HumanAgents" can access a ListViewPage/Component in the frontend that contains a list of assignments that have been assigned to them, from this list page they can access the details view of a single assignment and then from here they can check the status of the assignment and other details. Remember that  The assignments are supposed to be delivered through the chat. it is from the details page that the "ChatPage" of the particular assignment will be accessible from. So before an Assignment's "60%" timeline has been reached, then there will be a warning on the Chat Page for the  Human Agent not to Submit the Assignment. Also before the TimeLine has been reached, the HumanAgent Cannot access the Assignment Submission and the Submission files of the selected assignment. This is why I started my Mentioning the Assignment List and Detail Pages in the start,  because it is from the Details page that the Human Agent will have access to the Submission files depending on whether the 60% timeline rule has been reached. And there will be A warning or you can call it a reminder on the Details page and the Chat Page depending on whether the 60% rule has been reached so that the human Agent can be reminded that the Assignment Should Not be submitted to the client. So While you implement the 60% rule functionality, Keep this in mind. Also Remember the Langchain Agent that will be implemented inside the main_agent_two.py file will be working with the HumanAgent in the ChatPage, Here is where instead of the message being sent dirrectly to the Freelancing Platform, it is intercepted by the LanGchain Agent inside main_agent_two.py file so that it can offer improvements and corrections. There should be a dedicated endpoint for this. When Implementing this Reach the part of the AI Do not continue with the part of integrating it with Upwork or any other freelancing platform. so the final message that has been agreed upon by the Human Agent and the LLM agent in agent_two is recorded in the database. Leave it at that at this point. it is from the DB that we will create the custom logic of sending the messages back to the original freelancing platform. we will implement the logic for each platform. So I want you to work on this while you implement the 60% rule of time/ deadlines. I hope you understand what I mean

'''
