import logging
from typing import Dict, Any
from django.db import transaction
from assignments.models import Assignment, AssignmentSubmission
from .main_agent_three import handle_assignment as agent_three_handle
from .main_agent_four import handle_request as agent_four_handle
from .main_agent_five import academic_writing_pipeline
from .main_agent_six import ReviewAgent

logger = logging.getLogger(__name__)

async def handle_completed_assignment(assignment_id: int) -> Dict[str, Any]:
    """
    Handle assignment after initial completion by Agent 3/4/5.
    Routes to Agent 6 for review and manages the feedback loop.
    """
    try:
        # Get assignment
        assignment = await Assignment.objects.aget(pk=assignment_id)
        
        # Get latest submission
        submission = await AssignmentSubmission.objects.filter(
            assignment=assignment
        ).latest('date_completed')

        # Initialize Agent 6 for review
        review_agent = ReviewAgent()
        
        # Review assignment
        review_result = await review_agent.review_assignment(
            assignment_id,
            submission.id
        )

        return {
            "status": "success",
            "assignment_id": assignment_id,
            "passed_review": review_result["passed"],
            "needs_revision": not review_result["passed"],
            "review_details": review_result
        }

    except Exception as e:
        logger.error(f"Error handling completed assignment: {str(e)}")
        raise
