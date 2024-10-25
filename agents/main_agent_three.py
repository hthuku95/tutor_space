
import logging
import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from django.core.files import File
from django.core.files.base import ContentFile
from django.conf import settings
from django.utils import timezone
import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from assignments.models import (
    Assignment,
    AssignmentSubmission,
    RevisionFile,
    File as AssignmentFile
)
from .main_agent_four import handle_request as handle_programming_request
from .main_agent_five import academic_writing_pipeline
from .projects.protocols.agent_protocol import (
    ServiceRequest,
    MessageType
)

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class ExecutionError(Exception):
    """Custom exception for execution errors"""
    pass

class FileHandlingError(Exception):
    """Custom exception for file handling errors"""
    pass

class PlanningAgent:
    """Agent 3 - Planning and Setup Agent"""

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4")
        self.parser = StrOutputParser()
        self.submission_base_path = Path(settings.MEDIA_ROOT) / 'submissions'
        self.submission_base_path.mkdir(exist_ok=True)

    async def handle_assignment(self, assignment_id: int) -> Dict[str, Any]:
        """Handle a new assignment after deposit payment"""
        try:
            # Validate assignment
            assignment = await self._validate_assignment(assignment_id)

            # Start logging for this assignment
            logger.info(f"Starting assignment handling for ID: {assignment_id}")
            
            # Create assignment directory
            assignment_dir = self._create_assignment_directory(assignment)

            # Analyze requirements with validation
            requirements_analysis = await self._analyze_requirements(assignment)
            if not self._validate_analysis(requirements_analysis):
                raise ValidationError("Invalid requirements analysis")

            # Create and validate execution plan
            execution_plan = await self._create_execution_plan(
                assignment,
                requirements_analysis
            )
            if not self._validate_execution_plan(execution_plan):
                raise ValidationError("Invalid execution plan")

            # Execute plan with proper error handling
            result = await self._execute_plan(
                assignment,
                execution_plan,
                assignment_dir
            )

            # Log successful completion
            logger.info(f"Successfully completed assignment {assignment_id}")

            return {
                "status": "success",
                "assignment_id": assignment_id,
                "execution_plan": execution_plan,
                "result": result
            }

        except ValidationError as e:
            logger.error(f"Validation error for assignment {assignment_id}: {str(e)}")
            await self._handle_failure(assignment_id, str(e), "validation")
            raise

        except ExecutionError as e:
            logger.error(f"Execution error for assignment {assignment_id}: {str(e)}")
            await self._handle_failure(assignment_id, str(e), "execution")
            raise

        except Exception as e:
            logger.error(f"Unexpected error for assignment {assignment_id}: {str(e)}")
            await self._handle_failure(assignment_id, str(e), "unexpected")
            raise

    async def _validate_assignment(self, assignment_id: int) -> Assignment:
        """Validate assignment and its state"""
        try:
            assignment = Assignment.objects.get(pk=assignment_id)
        except Assignment.DoesNotExist:
            raise ValidationError(f"Assignment {assignment_id} not found")

        if not assignment.has_deposit_been_paid:
            raise ValidationError("Deposit must be paid before planning can begin")

        if assignment.completed:
            raise ValidationError("Assignment is already marked as completed")

        if not assignment.assignment_type in ['P', 'A']:
            raise ValidationError(f"Invalid assignment type: {assignment.assignment_type}")

        if assignment.completion_deadline < timezone.now():
            raise ValidationError("Assignment deadline has already passed")

        return assignment

    def _create_assignment_directory(self, assignment: Assignment) -> Path:
        """Create and return assignment directory path"""
        try:
            assignment_dir = self.submission_base_path / f"assignment_{assignment.id}"
            assignment_dir.mkdir(parents=True, exist_ok=True)
            return assignment_dir
        except Exception as e:
            raise FileHandlingError(f"Failed to create assignment directory: {str(e)}")

    async def _analyze_requirements(self, assignment: Assignment) -> Dict[str, Any]:
        """Analyze assignment requirements with enhanced validation"""
        required_fields = {
            "type": str,
            "complexity": str,
            "required_skills": list,
            "deliverables": list,
            "quality_criteria": dict
        }

        try:
            analysis = await super()._analyze_requirements(assignment)
            
            # Validate analysis structure
            for field, field_type in required_fields.items():
                if field not in analysis:
                    raise ValidationError(f"Missing required field: {field}")
                if not isinstance(analysis[field], field_type):
                    raise ValidationError(
                        f"Invalid type for {field}: expected {field_type}, got {type(analysis[field])}"
                    )

            return analysis

        except Exception as e:
            raise ValidationError(f"Requirements analysis failed: {str(e)}")

    def _validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Validate the requirements analysis"""
        try:
            # Check for required sections
            required_sections = [
                "type", "complexity", "required_skills",
                "deliverables", "quality_criteria"
            ]
            if not all(section in analysis for section in required_sections):
                return False

            # Validate complexity level
            if analysis["complexity"] not in ["low", "medium", "high"]:
                return False

            # Validate deliverables
            if not analysis["deliverables"] or not isinstance(analysis["deliverables"], list):
                return False

            # Validate quality criteria
            if not analysis["quality_criteria"] or not isinstance(analysis["quality_criteria"], dict):
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating analysis: {str(e)}")
            return False

    def _validate_execution_plan(self, plan: Dict[str, Any]) -> bool:
        """Validate the execution plan"""
        try:
            # Check for required sections
            required_sections = [
                "steps", "deliverables", "timeline",
                "resources", "validation_points"
            ]
            if not all(section in plan for section in required_sections):
                return False

            # Validate steps
            if not plan["steps"] or not isinstance(plan["steps"], list):
                return False

            # Validate timeline
            if not isinstance(plan["timeline"], dict):
                return False

            # Check for critical fields in each step
            for step in plan["steps"]:
                if not all(field in step for field in ["id", "name", "description", "output"]):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating execution plan: {str(e)}")
            return False

    async def _handle_programming_task(
        self,
        assignment: Assignment,
        execution_plan: Dict[str, Any],
        assignment_dir: Path
    ) -> Dict[str, Any]:
        """Handle programming task with enhanced file handling"""
        try:
            # Create service request for Agent 4
            request = await self._create_programming_request(
                assignment,
                execution_plan
            )

            # Send to Agent 4
            result = await handle_programming_request(request)

            # Validate the result
            if not self._validate_programming_result(result):
                raise ExecutionError("Invalid programming task result")

            # Handle generated files
            processed_files = await self._process_programming_files(
                result["files"],
                assignment_dir
            )

            # Create assignment submission
            submission = await self._create_submission(
                assignment,
                result,
                "programming",
                processed_files
            )

            return {
                "status": "completed",
                "submission_id": submission.id,
                "files": processed_files
            }

        except Exception as e:
            raise ExecutionError(f"Programming task handling failed: {str(e)}")

    async def _process_programming_files(
        self,
        files: Dict[str, str],
        assignment_dir: Path
    ) -> List[Dict[str, Any]]:
        """Process and save programming files"""
        processed_files = []

        try:
            for file_path, content in files.items():
                # Create full path
                full_path = assignment_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)

                # Save file
                with open(full_path, 'w') as f:
                    f.write(content)

                # Create assignment file record
                file_record = await self._create_file_record(
                    full_path,
                    "programming",
                    file_path
                )
                processed_files.append({
                    "path": file_path,
                    "file_id": file_record.id
                })

            return processed_files

        except Exception as e:
            raise FileHandlingError(f"File processing failed: {str(e)}")

    async def _handle_academic_writing(
        self,
        assignment: Assignment,
        execution_plan: Dict[str, Any],
        assignment_dir: Path
    ) -> Dict[str, Any]:
        """Handle academic writing with enhanced file handling"""
        try:
            # Extract parameters from execution plan
            word_count = execution_plan.get("word_count", 1000)
            file_name = f"assignment_{assignment.id}.docx"
            full_path = assignment_dir / file_name

            # Execute academic writing pipeline
            content_store, doc, generated_file = await academic_writing_pipeline(
                question=assignment.description,
                word_count=word_count,
                docx_file_name=str(full_path)
            )

            # Validate generated content
            if not self._validate_academic_content(doc, word_count):
                raise ValidationError("Generated content does not meet requirements")

            # Create file record
            file_record = await self._create_file_record(
                full_path,
                "academic_writing",
                file_name
            )

            # Create submission
            submission = await self._create_submission(
                assignment,
                {"document": doc, "content_store": content_store},
                "academic_writing",
                [{"path": file_name, "file_id": file_record.id}]
            )

            return {
                "status": "completed",
                "submission_id": submission.id,
                "file_name": file_name
            }

        except Exception as e:
            raise ExecutionError(f"Academic writing task failed: {str(e)}")

    async def _create_file_record(
        self,
        file_path: Path,
        file_type: str,
        original_name: str
    ) -> AssignmentFile:
        """Create a file record in the database"""
        try:
            with open(file_path, 'rb') as f:
                file_obj = File(f)
                assignment_file = AssignmentFile.objects.create(
                    file=file_obj,
                    origin=f"{file_type}_submission"
                )
                assignment_file.file.name = str(file_path.relative_to(settings.MEDIA_ROOT))
                assignment_file.save()
                return assignment_file

        except Exception as e:
            raise FileHandlingError(f"File record creation failed: {str(e)}")

    async def _create_submission(
        self,
        assignment: Assignment,
        result: Dict[str, Any],
        submission_type: str,
        files: List[Dict[str, Any]]
    ) -> AssignmentSubmission:
        """Create submission with proper file handling"""
        try:
            # Calculate delivery date
            delivery_date = self._calculate_delivery_date(assignment)

            # Create submission record
            submission = AssignmentSubmission.objects.create(
                assignment=assignment,
                date_completed=timezone.now(),
                date_to_be_delivered=delivery_date,
                version="1.0"
            )

            # Create revision file and attach files
            revision_file = RevisionFile.objects.create()
            
            # Add files to revision file
            for file_info in files:
                file_obj = AssignmentFile.objects.get(pk=file_info["file_id"])
                revision_file.files.add(file_obj)

            # Add revision file to submission
            submission.files.add(revision_file)

            return submission

        except Exception as e:
            raise FileHandlingError(f"Submission creation failed: {str(e)}")

    def _calculate_delivery_date(self, assignment: Assignment) -> datetime.datetime:
        """Calculate delivery date based on 60% rule"""
        now = timezone.now()
        time_until_deadline = assignment.completion_deadline - now
        delivery_delay = datetime.timedelta(
            seconds=time_until_deadline.total_seconds() * 0.6
        )
        return now + delivery_delay

    async def _handle_failure(
        self,
        assignment_id: int,
        error_message: str,
        error_type: str
    ):
        """Handle assignment failure"""
        try:
            assignment = Assignment.objects.get(pk=assignment_id)
            
            # Log error
            logger.error(
                f"Assignment {assignment_id} failed - Type: {error_type}, Error: {error_message}"
            )

            # Create failure record or notification
            # Implementation depends on your error handling requirements

        except Exception as e:
            logger.error(f"Error handling failure for assignment {assignment_id}: {str(e)}")

# Initialize the agent
planning_agent = PlanningAgent()

# Function to handle assignments
async def handle_assignment(assignment_id: int) -> Dict[str, Any]:
    """Handle assignment planning and execution"""
    return await planning_agent.handle_assignment(assignment_id)
