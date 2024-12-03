import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timedelta
import aiofiles
from django.core.files.base import ContentFile 

from django.conf import settings
from django.utils import timezone
from django.db import transaction
from django.core.files import File
from asgiref.sync import sync_to_async  

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from assignments.models import (
    Assignment, 
    AssignmentSubmission,  
    AssignmentFile,  
    RevisionFile  
)

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from .main_agent_four import handle_request as handle_programming_request
from .main_agent_five import academic_writing_pipeline

from .protocols.agent_protocol import (
    ServiceRequest,
    MessageType
)

from containers.utils import DockerClientManager


logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(
        self, 
        message: str, 
        validation_type: str = None,
        field: str = None,
        details: Dict = None
    ):
        self.validation_type = validation_type
        self.field = field
        self.details = details or {}
        self.message = self._format_message(message)
        self.docker_manager = DockerClientManager()
        
        super().__init__(self.message)

    def _format_message(self, message: str) -> str:
        parts = [message]
        if self.validation_type:
            parts.append(f"Validation Type: {self.validation_type}")
        if self.field:
            parts.append(f"Field: {self.field}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)

class ExecutionError(Exception):
    """Custom exception for execution errors"""
    def __init__(
        self,
        message: str,
        task_type: str = None,
        stage: str = None,
        error_code: str = None
    ):
        self.task_type = task_type
        self.stage = stage
        self.error_code = error_code
        self.message = self._format_message(message)
        super().__init__(self.message)

    def _format_message(self, message: str) -> str:
        parts = [message]
        if self.task_type:
            parts.append(f"Task Type: {self.task_type}")
        if self.stage:
            parts.append(f"Stage: {self.stage}")
        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")
        return " | ".join(parts)

class FileHandlingError(Exception):
    """Custom exception for file handling errors"""
    def __init__(
        self,
        message: str,
        operation: str = None,
        file_path: str = None,
        file_type: str = None,
        os_error: Exception = None
    ):
        self.operation = operation
        self.file_path = file_path
        self.file_type = file_type
        self.os_error = os_error
        self.message = self._format_message(message)
        super().__init__(self.message)

    def _format_message(self, message: str) -> str:
        parts = [message]
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        if self.file_path:
            parts.append(f"File Path: {self.file_path}")
        if self.file_type:
            parts.append(f"File Type: {self.file_type}")
        if self.os_error:
            parts.append(f"OS Error: {str(self.os_error)}")
        return " | ".join(parts)

class PlanningError(Exception):
    """Custom exception for planning errors"""
    def __init__(
        self,
        message: str,
        plan_stage: str = None,
        missing_requirements: List[str] = None,
        invalid_fields: List[str] = None
    ):
        self.plan_stage = plan_stage
        self.missing_requirements = missing_requirements or []
        self.invalid_fields = invalid_fields or []
        self.message = self._format_message(message)
        self.docker_manager = DockerClientManager()
        self.parser = JsonOutputParser()
        super().__init__(self.message)

    def _format_message(self, message: str) -> str:
        parts = [message]
        if self.plan_stage:
            parts.append(f"Planning Stage: {self.plan_stage}")
        if self.missing_requirements:
            parts.append(f"Missing Requirements: {', '.join(self.missing_requirements)}")
        if self.invalid_fields:
            parts.append(f"Invalid Fields: {', '.join(self.invalid_fields)}")
        return " | ".join(parts)
    
class AgentInitializationError(Exception):
    """Exception raised when Agent 3 fails to initialize"""
    def __init__(self, message: str, reason: str = None, details: Dict = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(message)

class PlanningAgent:
    """
    Agent 3 - Planning and Setup Agent
    Responsible for:
    1. Analyzing assignment requirements
    2. Creating execution plans
    3. Coordinating with Agent 4 (Programming) and Agent 5 (Academic Writing)
    4. Managing file submissions
    5. Implementing the 60% time rule
    """

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4")
        self.parser = JsonOutputParser()
        self.submission_base_path = Path(settings.MEDIA_ROOT) / 'submissions'
        self.submission_base_path.mkdir(exist_ok=True)
        self.required_analysis_fields = {
            "type": str,
            "complexity": ["low", "medium", "high"],
            "required_skills": list,
            "deliverables": list,
            "quality_criteria": dict,
            "estimated_effort": dict,
            "potential_challenges": list
        }
        self.required_plan_fields = {
            "steps": list,
            "deliverables": list,
            "timeline": dict,
            "resources": list,
            "validation_points": list,
            "quality_checks": dict,
            "contingency_plans": list
        }
        self.docker_manager = DockerClientManager()
        self.embeddings = OpenAIEmbeddings()

    async def handle_assignment(self, assignment_id: int) -> Dict[str, Any]:
        """Main entry point for handling assignments"""
        assignment = None
        assignment_dir = None
        try:
            # Start logging
            logger.info(f"Starting assignment handling for ID: {assignment_id}")
            
            # Validate assignment
            assignment = await self._validate_assignment(assignment_id)
            
            # Create assignment directory
            assignment_dir = self._create_assignment_directory(assignment)
            
            # Create vector store from assignment details
            vector_store = await self._create_assignment_vector_store(assignment)
            
            # Begin transaction
            with transaction.atomic():
                # Analyze requirements using vector store for context
                requirements_analysis = await self._analyze_requirements(
                    assignment,
                    vector_store=vector_store
                )
                
                # Create execution plan based on analysis
                execution_plan = await self._create_execution_plan(
                    assignment,
                    requirements_analysis
                )
                
                # Execute plan
                logger.info("Executing plan...")
                result = await self._execute_plan(
                    assignment,
                    execution_plan,
                    assignment_dir
                )
                
                # Update assignment status
                await self._notify_status_change(
                    assignment,
                    "in_progress",
                    execution_plan
                )
                
                # Create final response
                return {
                    "status": "success",
                    "assignment_id": assignment_id,
                    "analysis": requirements_analysis,
                    "execution_plan": execution_plan,
                    "result": result
                }

        except ValidationError as e:
            error_msg = f"Validation error for assignment {assignment_id}: {str(e)}"
            logger.error(error_msg)
            await self._notify_failure(assignment, error_msg, "validation")
            raise
            
        except ExecutionError as e:
            error_msg = f"Execution error for assignment {assignment_id}: {str(e)}"
            logger.error(error_msg)
            await self._notify_failure(assignment, error_msg, "execution")
            raise
            
        except Exception as e:
            error_msg = f"Unexpected error for assignment {assignment_id}: {str(e)}"
            logger.error(error_msg)
            await self._notify_failure(assignment, error_msg, "unexpected")
            raise
            
        finally:
            if assignment_dir and assignment_dir.exists():
                await self._cleanup_temp_files(assignment_dir)


    async def _analyze_requirements(
        self,
        assignment: Assignment,
        vector_store: FAISS
    ) -> Dict[str, Any]:
        """Analyze requirements with vector store context"""
        # Get relevant information from vector store
        query = f"What are the key requirements and constraints for this {assignment.get_assignment_type_display()}?"
        relevant_docs = await sync_to_async(vector_store.similarity_search)(
            query,
            k=5  # Get top 5 most relevant chunks
        )
        
        context = "\n".join(doc.page_content for doc in relevant_docs)
        
        # Use context in prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the assignment requirements using the provided context.
            Consider all details, files, and constraints mentioned."""),
            ("human", """
            Assignment to analyze:
            {assignment_description}
            
            Additional Context:
            {context}
            
            Provide a detailed analysis following the required format.
            """)
        ])

        try:
            result = await (prompt | self.model | self.parser).ainvoke({
                "assignment_description": assignment.description,
                "context": context
            })
            
            return result

        except Exception as e:
            logger.error(f"Error in requirements analysis: {str(e)}")
            raise

    def _create_assignment_directory(self, assignment: Assignment) -> Path:
        """Create and return assignment directory path"""
        try:
            assignment_dir = self.submission_base_path / f"assignment_{assignment.id}"
            assignment_dir.mkdir(parents=True, exist_ok=True)
            return assignment_dir
        except Exception as e:
            raise FileHandlingError(
                "Failed to create assignment directory",
                operation="create_directory",
                file_path=str(assignment_dir),
                os_error=e
            )

    async def _cleanup_temp_files(self, directory: Path) -> None:
        """Clean up temporary files"""
        try:
            if directory.exists():
                for file_path in directory.glob("**/*"):
                    if file_path.is_file() and file_path.name.startswith('temp_'):
                        file_path.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")

    async def _notify_failure(
        self,
        assignment: Assignment,
        error_message: str,
        error_type: str
    ) -> None:
        """Handle assignment failure"""
        try:
            if assignment:
                # Log the failure
                logger.error(
                    f"Assignment {assignment.id} failed - "
                    f"Type: {error_type}, Error: {error_message}"
                )

                # Update assignment status
                assignment.completed = False
                assignment.has_revisions = False
                assignment.last_error = json.dumps({
                    "type": error_type,
                    "message": error_message,
                    "timestamp": timezone.now().isoformat()
                })
                await self._save_assignment(assignment)

        except Exception as e:
            logger.error(f"Error handling failure notification: {str(e)}")

    async def _notify_status_change(
        self,
        assignment: Assignment,
        status: str,
        execution_plan: Dict[str, Any]
    ) -> None:
        """Handle status change"""
        try:
            # Log the status change
            logger.info(
                f"Assignment {assignment.id} status changed to {status} - "
                f"Progress: {self._calculate_progress(execution_plan)}%"
            )

            # Update assignment based on status
            if status == "completed":
                assignment.completed = True
                assignment.has_revisions = False
            elif status == "in_progress":
                assignment.completed = False
                assignment.has_revisions = False
            elif status == "revision_needed":
                assignment.completed = False
                assignment.has_revisions = True

            assignment.last_updated = timezone.now()
            await self._save_assignment(assignment)

        except Exception as e:
            logger.error(f"Error handling status change: {str(e)}")

    def _calculate_progress(self, execution_plan: Dict[str, Any]) -> float:
        """Calculate current progress percentage"""
        try:
            completed_steps = sum(
                1 for step in execution_plan.get("steps", [])
                if step.get("completed")
            )
            total_steps = len(execution_plan.get("steps", []))
            return (completed_steps / total_steps) * 100 if total_steps > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating progress: {str(e)}")
            return 0.0

    async def _save_assignment(self, assignment: Assignment) -> None:
        """Save assignment with error handling"""
        try:
            await sync_to_async(assignment.save)()
        except Exception as e:
            logger.error(f"Error saving assignment: {str(e)}")
            raise

    async def _validate_assignment(self, assignment_id: int) -> Assignment:
        """Comprehensive assignment validation"""
        try:
            # Get assignment with related fields
            assignment = await sync_to_async(Assignment.objects.select_related(
                'agent', 'original_platform', 'original_account'
            ).get)(pk=assignment_id)

        except Assignment.DoesNotExist:
            raise ValidationError(
                f"Assignment {assignment_id} not found",
                validation_type="assignment",
                field="id"
            )

        # Validate assignment state
        if not assignment.has_deposit_been_paid:
            raise ValidationError(
                "Deposit must be paid before planning can begin",
                validation_type="assignment",
                field="has_deposit_been_paid"
            )

        if assignment.completed:
            raise ValidationError(
                "Assignment is already marked as completed",
                validation_type="assignment",
                field="completed"
            )

        if not assignment.assignment_type in ['P', 'A']:
            raise ValidationError(
                f"Invalid assignment type: {assignment.assignment_type}",
                validation_type="assignment",
                field="assignment_type"
            )

        if assignment.completion_deadline < timezone.now():
            raise ValidationError(
                "Assignment deadline has already passed",
                validation_type="assignment",
                field="completion_deadline"
            )

        # Validate required fields
        if not assignment.subject or not assignment.description:
            raise ValidationError(
                "Assignment missing required fields",
                validation_type="assignment",
                field="required_fields",
                details={"missing": [
                    f for f in ['subject', 'description']
                    if not getattr(assignment, f)
                ]}
            )

        # Validate assignment files if they exist
        if await sync_to_async(assignment.assignment_files.exists)():
            async for file in sync_to_async(list)(assignment.assignment_files.all()):
                if not await sync_to_async(file.files.exists)():
                    raise ValidationError(
                        f"Assignment file {file.id} is invalid",
                        validation_type="assignment",
                        field="files"
                    )

        return assignment

    async def _analyze_requirements(self, assignment: Assignment) -> Dict[str, Any]:
        """Analyze assignment requirements"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert project analyzer specializing in both programming and academic writing tasks.
            Analyze the given assignment requirements and provide a detailed analysis including:
            1. Type and complexity assessment
            2. Required skills and technologies
            3. Deliverable specifications
            4. Quality criteria
            5. Estimated effort
            6. Potential challenges and risks
            
            Return a structured JSON response with these components.
            The response must include exactly these fields:
            {
                "type": "string",
                "complexity": "low|medium|high",
                "required_skills": ["skill1", "skill2"],
                "deliverables": ["deliverable1", "deliverable2"],
                "quality_criteria": {"criterion1": "description1"},
                "estimated_effort": {"hours": number, "confidence": "low|medium|high"},
                "potential_challenges": ["challenge1", "challenge2"]
            }"""),
            ("human", """Analyze this assignment:
            Subject: {subject}
            Description: {description}
            Type: {assignment_type}
            Deadline: {deadline}
            Files: {files}
            
            Provide a comprehensive analysis following the specified format.""")
        ])

        try:
            # Prepare files information
            files_info = []
            if await sync_to_async(assignment.assignment_files.exists)():
                files = await sync_to_async(list)(assignment.assignment_files.all())
                files_info = [
                    {
                        "name": await sync_to_async(getattr)(f, 'name'),
                        "type": await sync_to_async(getattr)(
                            f, 'content_type', None
                        ) if hasattr(f, 'content_type') else None
                    }
                    for f in files
                ]

            # Get analysis from LLM
            analysis_str = await (prompt | self.model | StrOutputParser()).ainvoke({
                "subject": assignment.subject,
                "description": assignment.description,
                "assignment_type": (
                    "Programming" if assignment.assignment_type == "P"
                    else "Academic Writing"
                ),
                "deadline": assignment.completion_deadline.isoformat(),
                "files": json.dumps(files_info)
            })

            # Parse and validate analysis
            try:
                analysis = json.loads(analysis_str)
            except json.JSONDecodeError as e:
                raise ValidationError(
                    f"Invalid analysis format: {str(e)}",
                    validation_type="analysis",
                    field="format"
                )

            # Validate analysis structure
            await self._validate_analysis_structure(analysis)
            
            # Enhance analysis with additional context
            enhanced_analysis = await self._enhance_analysis(analysis, assignment)
            
            return enhanced_analysis

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Requirements analysis failed: {str(e)}",
                validation_type="analysis"
            )

    async def _validate_analysis_structure(self, analysis: Dict[str, Any]) -> None:
        """Validate the analysis structure"""
        for field, field_type in self.required_analysis_fields.items():
            if field not in analysis:
                raise ValidationError(
                    f"Missing required field in analysis: {field}",
                    validation_type="analysis",
                    field=field
                )
            
            if isinstance(field_type, list):
                if analysis[field] not in field_type:
                    raise ValidationError(
                        f"Invalid value for {field}: {analysis[field]}",
                        validation_type="analysis",
                        field=field
                    )
            elif not isinstance(analysis[field], field_type):
                raise ValidationError(
                    f"Invalid type for {field}",
                    validation_type="analysis",
                    field=field,
                    details={
                        "expected": str(field_type),
                        "got": str(type(analysis[field]))
                    }
                )
    
    async def handle_revision(
        self,
        assignment: Assignment,
        revision_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle revision request for an assignment
        This is called when Agent 6 determines revisions are needed or
        when client requests revisions
        """
        try:
            # Get latest submission
            latest_submission = await sync_to_async(
                assignment.submissions.latest
            )('date_completed')

            # Create revision record
            revision = await self._create_revision_record(
                assignment,
                latest_submission,
                revision_request
            )

            # Re-analyze requirements with revision context
            requirements_analysis = await self._analyze_requirements(
                assignment,
                revision_context=revision_request
            )

            # Create revision execution plan
            execution_plan = await self._create_execution_plan(
                assignment,
                requirements_analysis,
                is_revision=True
            )

            # Execute revision
            if assignment.assignment_type == "P":
                result = await self._handle_programming_revision(
                    assignment,
                    revision,
                    execution_plan
                )
            else:
                result = await self._handle_academic_writing_revision(
                    assignment,
                    revision,
                    execution_plan
                )

            return result

        except Exception as e:
            logger.error(f"Error handling revision: {str(e)}")
            raise

    async def _create_revision_record(
        self,
        assignment: Assignment,
        previous_submission: AssignmentSubmission,
        revision_request: Dict[str, Any]
    ) -> RevisionFile:
        """Create revision record in database"""
        try:
            return await sync_to_async(RevisionFile.objects.create)(
                assignment=assignment,
                assignment_submission=previous_submission,
                reason_for_submission=revision_request.get('reason', ''),
                deadline=timezone.now() + timedelta(
                    days=revision_request.get('deadline_days', 2)
                ),
                revision_price=revision_request.get('price', 0.0),
                needs_payment=revision_request.get('needs_payment', False)
            )
        except Exception as e:
            logger.error(f"Error creating revision record: {str(e)}")
            raise

    async def _handle_programming_revision(
        self,
        assignment: Assignment,
        revision: RevisionFile,
        execution_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle programming task revision"""
        try:
            # Update execution plan with revision context
            execution_plan["is_revision"] = True
            execution_plan["previous_submission"] = await self._get_previous_submission_info(
                revision.assignment_submission
            )

            # Create revision directory
            revision_dir = self._create_revision_directory(assignment, revision)

            # Process revision using Agent 4
            result = await self._handle_programming_task(
                assignment,
                execution_plan,
                revision_dir
            )

            # Update revision record with results
            await self._update_revision_record(revision, result)

            return result

        except Exception as e:
            logger.error(f"Error handling programming revision: {str(e)}")
            raise

    async def _handle_academic_writing_revision(
        self,
        assignment: Assignment,
        revision: RevisionFile,
        execution_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle academic writing revision"""
        try:
            # Similar to programming revision but using Agent 5
            execution_plan["is_revision"] = True
            execution_plan["previous_submission"] = await self._get_previous_submission_info(
                revision.assignment_submission
            )

            revision_dir = self._create_revision_directory(assignment, revision)

            result = await self._handle_academic_writing(
                assignment,
                execution_plan,
                revision_dir
            )

            await self._update_revision_record(revision, result)

            return result

        except Exception as e:
            logger.error(f"Error handling academic writing revision: {str(e)}")
            raise

    def _create_revision_directory(
        self,
        assignment: Assignment,
        revision: RevisionFile
    ) -> Path:
        """Create directory for revision files"""
        revision_dir = (
            self.submission_base_path / 
            f"assignment_{assignment.id}" /
            f"revision_{revision.id}"
        )
        revision_dir.mkdir(parents=True, exist_ok=True)
        return revision_dir

    async def _get_previous_submission_info(
        self,
        submission: AssignmentSubmission
    ) -> Dict[str, Any]:
        """Get information about previous submission for context"""
        files_info = []
        async for file in submission.files.aiterator():
            files_info.append({
                "path": file.file.name,
                "content": await self._read_file_content(file)
            })

        return {
            "submission_id": submission.id,
            "version": submission.version,
            "files": files_info,
            "date_completed": submission.date_completed.isoformat()
        }

    async def _update_revision_record(
        self,
        revision: RevisionFile,
        result: Dict[str, Any]
    ) -> None:
        """Update revision record with results"""
        try:
            revision.has_deposit_been_paid = result.get('payment_received', False)
            await sync_to_async(revision.save)()

            # Add files to revision
            for file_info in result.get('files', []):
                file_record = await self._create_file_record(
                    file_info['path'],
                    'revision',
                    file_info['name']
                )
                await sync_to_async(revision.revision_files.add)(file_record)

        except Exception as e:
            logger.error(f"Error updating revision record: {str(e)}")
            raise

    async def _create_assignment_vector_store(
        self,
        assignment: Assignment
    ) -> FAISS:
        """
        Create a vector store from assignment details to help LLMs understand the task.
        Includes assignment details, requirements, and any attached files.
        """
        try:
            # Collect assignment texts
            texts = []
            metadatas = []

            # Basic assignment details
            texts.append(f"Assignment Subject: {assignment.subject}")
            metadatas.append({"field": "subject", "type": "basic_info"})

            texts.append(f"Assignment Description: {assignment.description}")
            metadatas.append({"field": "description", "type": "basic_info"})

            texts.append(f"Assignment Type: {'Programming' if assignment.assignment_type == 'P' else 'Academic Writing'}")
            metadatas.append({"field": "type", "type": "basic_info"})

            # Time constraints
            deadline_info = (
                f"Deadline: {assignment.completion_deadline.isoformat()}, "
                f"Expected Delivery: {assignment.expected_delivery_time.isoformat()}"
            )
            texts.append(deadline_info)
            metadatas.append({"field": "timeline", "type": "time_constraints"})

            # Process assignment files if they exist
            if await sync_to_async(assignment.assignment_files.exists)():
                async for assignment_file in sync_to_async(list)(assignment.assignment_files.all()):
                    file_content = await self._extract_file_content(assignment_file)
                    if file_content:
                        texts.append(f"File Content ({assignment_file.file.name}): {file_content}")
                        metadatas.append({
                            "field": "file_content",
                            "type": "file",
                            "file_name": assignment_file.file.name
                        })

            # Create vector store
            vector_store = await self._create_vector_store(texts, metadatas)
            
            return vector_store

        except Exception as e:
            logger.error(f"Error creating assignment vector store: {str(e)}")
            raise

    async def _extract_file_content(self, assignment_file: AssignmentFile) -> Optional[str]:
        """Extract content from assignment file based on file type"""
        try:
            file_path = assignment_file.file.path
            file_extension = Path(file_path).suffix.lower()

            # Handle different file types
            if file_extension in ['.txt', '.py', '.js', '.html', '.css', '.java']:
                # Text files
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    return await f.read()
                    
            elif file_extension in ['.pdf', '.doc', '.docx']:
                # Use appropriate library based on file type
                # For example, use pypdf for PDFs
                return await self._extract_document_content(file_path, file_extension)
                
            else:
                logger.warning(f"Unsupported file type for content extraction: {file_extension}")
                return None

        except Exception as e:
            logger.error(f"Error extracting file content: {str(e)}")
            return None

    async def _create_vector_store(
        self,
        texts: List[str],
        metadatas: List[Dict[str, str]]
    ) -> FAISS:
        """Create FAISS vector store from texts and metadata"""
        try:
            # Create vector store
            vector_store = await sync_to_async(FAISS.from_texts)(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            return vector_store

        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    async def _extract_document_content(
        self,
        file_path: str,
        file_extension: str
    ) -> Optional[str]:
        """Extract content from document files"""
        try:
            if file_extension == '.pdf':
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
            elif file_extension in ['.doc', '.docx']:
                from docx import Document
                doc = Document(file_path)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
                
            return None

        except Exception as e:
            logger.error(f"Error extracting document content: {str(e)}")
            return None

    async def _enhance_analysis(
        self,
        analysis: Dict[str, Any],
        assignment: Assignment
    ) -> Dict[str, Any]:
        """Enhance analysis with additional context and validations"""
        # Add time-based complexity factors
        time_until_deadline = assignment.completion_deadline - timezone.now()
        analysis["time_constraints"] = {
            "days_available": time_until_deadline.days,
            "hours_available": time_until_deadline.seconds // 3600,
            "is_urgent": time_until_deadline.days < 2
        }

        # Add effort estimation details
        analysis["effort_distribution"] = await self._estimate_effort_distribution(
            analysis["estimated_effort"],
            analysis["complexity"],
            assignment.assignment_type
        )

        # Add risk levels
        analysis["risk_levels"] = await self._assess_risk_levels(
            analysis["potential_challenges"],
            time_until_deadline
        )

        return analysis

    async def _estimate_effort_distribution(
        self,
        estimated_effort: Dict[str, Any],
        complexity: str,
        assignment_type: str
    ) -> Dict[str, float]:
        """Estimate effort distribution across different phases"""
        total_hours = estimated_effort.get("hours", 0)
        
        if assignment_type == "P":  # Programming
            return {
                "planning": total_hours * 0.15,
                "development": total_hours * 0.5,
                "testing": total_hours * 0.25,
                "documentation": total_hours * 0.1
            }
        else:  # Academic Writing
            return {
                "research": total_hours * 0.3,
                "writing": total_hours * 0.4,
                "editing": total_hours * 0.2,
                "formatting": total_hours * 0.1
            }

    async def _assess_risk_levels(
        self,
        challenges: List[str],
        time_until_deadline: timedelta
    ) -> Dict[str, str]:
        """Assess risk levels for identified challenges"""
        risk_levels = {}
        
        for challenge in challenges:
            # Determine risk level based on challenge and time
            if time_until_deadline.days < 2:
                risk_level = "high"
            elif time_until_deadline.days < 5:
                risk_level = "medium"
            else:
                risk_level = "low"
                
            risk_levels[challenge] = risk_level
            
        return risk_levels
    
    async def _create_execution_plan(
        self,
        assignment: Assignment,
        requirements_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create detailed execution plan"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert project planner.
            Create a detailed execution plan that includes:
            1. Step-by-step breakdown of tasks
            2. Resource allocation
            3. Timeline and milestones
            4. Quality control measures
            5. Risk mitigation strategies
            6. Contingency plans
            
            The plan must include exactly these fields:
            {
                "steps": [
                    {
                        "id": "string",
                        "name": "string",
                        "description": "string",
                        "estimated_duration": number,
                        "dependencies": ["step_id"],
                        "outputs": ["output_description"]
                    }
                ],
                "deliverables": ["deliverable_description"],
                "timeline": {
                    "start": "ISO-date",
                    "end": "ISO-date",
                    "milestones": [
                        {
                            "name": "string",
                            "date": "ISO-date",
                            "criteria": ["criterion"]
                        }
                    ]
                },
                "resources": ["resource_description"],
                "validation_points": ["validation_description"],
                "quality_checks": {
                    "check_name": "check_description"
                },
                "contingency_plans": ["plan_description"]
            }"""),
            ("human", """Create an execution plan for:
            Assignment: {assignment_details}
            Analysis: {requirements_analysis}
            Time Constraints: {time_constraints}
            
            Return a detailed plan following the specified format.""")
        ])

        try:
            # Get plan from LLM
            plan_str = await (prompt | self.model | StrOutputParser()).ainvoke({
                "assignment_details": {
                    "subject": assignment.subject,
                    "description": assignment.description,
                    "type": assignment.assignment_type,
                    "deadline": assignment.completion_deadline.isoformat()
                },
                "requirements_analysis": requirements_analysis,
                "time_constraints": {
                    "delivery_target": self._calculate_delivery_date(assignment),
                    "total_available_time": (
                        assignment.completion_deadline - timezone.now()
                    ).total_seconds()
                }
            })

            # Parse and validate plan
            try:
                plan = json.loads(plan_str)
            except json.JSONDecodeError as e:
                raise PlanningError(
                    f"Invalid plan format: {str(e)}",
                    plan_stage="parsing"
                )

            # Validate plan structure
            await self._validate_plan_structure(plan)
            
            # Enhance plan with specific requirements
            enhanced_plan = await self._enhance_plan(
                plan,
                assignment,
                requirements_analysis
            )
            
            return enhanced_plan

        except PlanningError:
            raise
        except Exception as e:
            raise PlanningError(
                f"Plan creation failed: {str(e)}",
                plan_stage="creation"
            )

    async def _validate_plan_structure(self, plan: Dict[str, Any]) -> None:
        """Validate the plan structure"""
        for field, field_type in self.required_plan_fields.items():
            if field not in plan:
                raise PlanningError(
                    f"Missing required field in plan: {field}",
                    plan_stage="validation",
                    missing_requirements=[field]
                )
            
            if not isinstance(plan[field], field_type):
                raise PlanningError(
                    f"Invalid type for {field}",
                    plan_stage="validation",
                    invalid_fields=[field]
                )

        # Validate steps structure
        for step in plan["steps"]:
            required_step_fields = ["id", "name", "description", "estimated_duration"]
            missing_fields = [
                field for field in required_step_fields
                if field not in step
            ]
            if missing_fields:
                raise PlanningError(
                    "Invalid step structure",
                    plan_stage="validation",
                    missing_requirements=missing_fields
                )

    async def _enhance_plan(
        self,
        plan: Dict[str, Any],
        assignment: Assignment,
        requirements_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance plan with specific requirements"""
        try:
            # Add detailed timeline
            plan["timeline"] = await self._create_detailed_timeline(
                plan["steps"],
                assignment.completion_deadline
            )
            
            # Add quality requirements
            plan["quality_requirements"] = await self._create_quality_requirements(
                requirements_analysis["quality_criteria"],
                assignment.assignment_type
            )
            
            # Add monitoring points
            plan["monitoring_points"] = await self._create_monitoring_points(
                plan["steps"]
            )

            return plan

        except Exception as e:
            raise PlanningError(
                f"Plan enhancement failed: {str(e)}",
                plan_stage="enhancement"
            )

    async def _create_detailed_timeline(
        self,
        steps: List[Dict[str, Any]],
        deadline: datetime
    ) -> Dict[str, Any]:
        """Create detailed timeline with milestones"""
        total_duration = sum(step["estimated_duration"] for step in steps)
        start_time = timezone.now()
        time_per_unit = (deadline - start_time).total_seconds() / total_duration

        timeline = {
            "start": start_time.isoformat(),
            "end": deadline.isoformat(),
            "milestones": []
        }

        current_time = start_time
        for step in steps:
            step_duration = timedelta(hours=step["estimated_duration"])
            milestone_time = current_time + step_duration
            
            timeline["milestones"].append({
                "name": f"Complete {step['name']}",
                "date": milestone_time.isoformat(),
                "criteria": [f"Completed: {output}" for output in step.get("outputs", [])]
            })
            
            current_time = milestone_time

        return timeline

    async def _create_quality_requirements(
        self,
        quality_criteria: Dict[str, str],
        assignment_type: str
    ) -> Dict[str, Any]:
        """Create detailed quality requirements"""
        base_requirements = {
            "documentation": "Complete and clear documentation",
            "testing": "Comprehensive testing coverage",
            "standards": "Adherence to coding/writing standards"
        }

        if assignment_type == "P":
            base_requirements.update({
                "code_quality": "Clean, maintainable code",
                "performance": "Efficient resource usage",
                "security": "Basic security measures implemented"
            })
        else:
            base_requirements.update({
                "originality": "No plagiarism",
                "citations": "Proper citations and references",
                "formatting": "Correct academic formatting"
            })

        # Merge with provided quality criteria
        return {**base_requirements, **quality_criteria}

    async def _create_monitoring_points(
        self,
        steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create monitoring points for progress tracking"""
        monitoring_points = []
        completion_percentage = 0
        step_percentage = 100 / len(steps)

        for step in steps:
            completion_percentage += step_percentage
            monitoring_points.append({
                "point": f"Complete {step['name']}",
                "completion_percentage": round(completion_percentage, 2),
                "verification": step.get("outputs", []),
                "dependencies": step.get("dependencies", [])
            })

        return monitoring_points
    
    async def _execute_plan(
        self,
        assignment: Assignment,
        execution_plan: Dict[str, Any],
        assignment_dir: Path
        ) -> Dict[str, Any]:
        """Execute the plan using appropriate agent"""
        try:
            if assignment.assignment_type == "P":
                logger.info(f"Handling programming task for assignment {assignment.id}")
                return await self._handle_programming_task(
                    assignment,
                    execution_plan,
                    assignment_dir
                )
            elif assignment.assignment_type == "A":
                logger.info(f"Handling academic writing task for assignment {assignment.id}")
                return await self._handle_academic_writing(
                    assignment,
                    execution_plan,
                    assignment_dir
                )
            else:
                raise ExecutionError(
                    f"Unsupported assignment type: {assignment.assignment_type}",
                    task_type=assignment.assignment_type,
                    stage="task_selection"
                )

        except Exception as e:
            raise ExecutionError(
                f"Plan execution failed: {str(e)}",
                task_type=assignment.assignment_type,
                stage="execution"
            )

    async def _handle_programming_task(
        self,
        assignment: Assignment,
        execution_plan: Dict[str, Any],
        assignment_dir: Path
    ) -> Dict[str, Any]:
        """
        Handle programming task through Agent 4.
        Includes code generation, Docker containerization, and validation.
        """
        container_ids = []
        try:
            # Create request for Agent 4
            logger.info(f"Creating request for Agent 4 for assignment {assignment.id}")
            service_request = await self._create_programming_request(
                assignment,
                execution_plan
            )

            # Send to Agent 4 for code generation
            logger.info("Sending request to Agent 4 for code generation")
            agent4_result = await handle_programming_request(service_request)

            # Validate Agent 4 result
            logger.info("Validating Agent 4 response")
            if not await self._validate_programming_result(agent4_result):
                raise ExecutionError(
                    "Invalid programming task result from Agent 4",
                    task_type="programming",
                    stage="validation"
                )

            # Process generated files
            logger.info("Processing generated files")
            processed_files = await self._process_programming_files(
                agent4_result["files"],
                assignment_dir
            )

            # Handle Docker containerization
            logger.info("Setting up Docker container")
            container_info = await self._handle_container_operations(
                assignment,
                agent4_result["files"],
                execution_plan
            )
            if container_info['container']['id']:
                container_ids.append(container_info['container']['id'])

            # Validate container setup
            logger.info("Validating container setup")
            if not await self._validate_container(
                container_info['container']['id'],
                execution_plan.get('validation_criteria', {})
            ):
                raise ExecutionError(
                    "Container validation failed",
                    task_type="programming",
                    stage="container_validation"
                )

            # Create assignment submission
            logger.info("Creating assignment submission")
            submission = await self._create_submission(
                assignment,
                {
                    **agent4_result,
                    "container_info": container_info
                },
                "programming",
                processed_files
            )

            # Prepare final result
            result = {
                "status": "completed",
                "submission_id": submission.id,
                "files": processed_files,
                "container_info": {
                    "image_id": container_info['image']['id'],
                    "image_tag": container_info['image']['tag'],
                    "container_id": container_info['container']['id'],
                    "port_mappings": container_info['container'].get('port_mappings', {}),
                    "access_urls": self._generate_access_urls(
                        container_info['container'].get('port_mappings', {})
                    ),
                    "container_status": container_info['container'].get('status')
                },
                "validation_results": agent4_result.get("validation_results", {}),
                "execution_metrics": {
                    "generation_time": agent4_result.get("generation_time"),
                    "build_time": container_info.get("build_time"),
                    "total_time": container_info.get("total_time")
                }
            }

            logger.info(f"Successfully completed programming task for assignment {assignment.id}")
            return result

        except ExecutionError:
            raise
        except Exception as e:
            error_msg = f"Error in programming task: {str(e)}"
            logger.error(error_msg)
            raise ExecutionError(
                error_msg,
                task_type="programming",
                stage="execution"
            )
        finally:
            # Cleanup containers if any were created
            if container_ids:
                logger.info(f"Cleaning up containers: {container_ids}")
                await self._cleanup_containers(
                    assignment.id,
                    container_ids
                )

        def _generate_access_urls(
            self,
            port_mappings: Dict[str, str]
        ) -> Dict[str, str]:
            """Generate access URLs for mapped ports"""
            base_url = settings.DOCKER_HOST or 'localhost'
            return {
                f"port_{container_port}": f"http://{base_url}:{host_port}"
                for container_port, host_port in port_mappings.items()
            }

    async def _handle_academic_writing(
        self,
        assignment: Assignment,
        execution_plan: Dict[str, Any],
        assignment_dir: Path
    ) -> Dict[str, Any]:
        """Handle academic writing task through Agent 5"""
        try:
            # Extract parameters from execution plan
            word_count = execution_plan.get("deliverables", {}).get(
                "word_count",
                self._calculate_word_count(assignment.description)
            )
            docx_file_name = f"assignment_{assignment.id}.docx"
            full_path = assignment_dir / docx_file_name

            # Execute academic writing pipeline
            logger.info(f"Starting academic writing pipeline for assignment {assignment.id}")
            content_store, doc, file_name = await academic_writing_pipeline(
                question=assignment.description,
                word_count=word_count,
                docx_file_name=str(full_path)
            )

            # Validate generated content
            logger.info("Validating generated content")
            if not await self._validate_academic_content(doc, word_count):
                raise ExecutionError(
                    "Generated content does not meet requirements",
                    task_type="academic_writing",
                    stage="validation"
                )

            # Create file record
            logger.info("Creating file record")
            file_record = await self._create_file_record(
                full_path,
                "academic_writing",
                docx_file_name
            )

            # Create submission
            logger.info("Creating submission")
            submission = await self._create_submission(
                assignment,
                {
                    "document": doc,
                    "content_store": content_store
                },
                "academic_writing",
                [{"path": docx_file_name, "file_id": file_record.id}]
            )

            return {
                "status": "completed",
                "submission_id": submission.id,
                "file_name": docx_file_name
            }

        except Exception as e:
            logger.error(f"Error in academic writing task: {str(e)}")
            raise ExecutionError(
                f"Academic writing task failed: {str(e)}",
                task_type="academic_writing",
                stage="execution"
            )

    async def _create_programming_request(
        self,
        assignment: Assignment,
        execution_plan: Dict[str, Any]
    ) -> ServiceRequest:
        """Create service request for Agent 4 with Docker specifications"""
        return ServiceRequest(
            request_id=str(assignment.id),
            message_type=MessageType.SERVICE_SPEC,
            service_spec={
                "assignment_details": {
                    "id": assignment.id,
                    "subject": assignment.subject,
                    "description": assignment.description,
                    "deadline": assignment.completion_deadline.isoformat(),
                    "type": "programming"
                },
                "execution_plan": execution_plan,
                "requirements": {
                    "optimization_targets": execution_plan.get("optimization_targets", []),
                    "quality_requirements": execution_plan.get("quality_checks", {}),
                    "timeline": execution_plan.get("timeline", {}),
                    "docker_requirements": {
                        "need_containerization": True,
                        "expose_ports": True,
                        "base_image_requirements": self._determine_base_image(assignment.description),
                        "environment_variables": self._extract_env_requirements(execution_plan),
                        "network_requirements": self._determine_network_requirements(execution_plan)
                    }
                }
            }
        )

    async def _validate_programming_result(self, result: Dict[str, Any]) -> bool:
        """Validate programming task result"""
        try:
            # Check required fields
            required_fields = ["files", "status", "validation_results"]
            if not all(field in result for field in required_fields):
                logger.error(f"Missing required fields in result: {required_fields}")
                return False

            # Check status
            if result["status"] != "completed":
                logger.error(f"Invalid status: {result['status']}")
                return False

            # Check files
            if not result["files"] or not isinstance(result["files"], dict):
                logger.error("Invalid files format")
                return False

            # Check validation results
            validation = result["validation_results"]
            if not validation.get("tests_passed"):
                logger.error("Tests not passed")
                return False
            if validation.get("critical_issues", []):
                logger.error(f"Critical issues found: {validation['critical_issues']}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating programming result: {str(e)}")
            return False

    async def _validate_academic_content(self, doc: Any, word_count: int) -> bool:
        """Validate academic writing content"""
        try:
            # Get text content
            text = doc.get_text() if hasattr(doc, 'get_text') else str(doc)
            actual_word_count = len(text.split())

            # Check word count (allow 10% deviation)
            word_count_margin = word_count * 0.1
            if abs(actual_word_count - word_count) > word_count_margin:
                logger.error(
                    f"Word count mismatch: expected {word_count}, "
                    f"got {actual_word_count}"
                )
                return False

            # Check for minimum sections
            required_sections = ['introduction', 'body', 'conclusion']
            text_lower = text.lower()
            if not all(section in text_lower for section in required_sections):
                missing_sections = [
                    s for s in required_sections if s not in text_lower
                ]
                logger.error(f"Missing sections: {missing_sections}")
                return False

            # Check for citations
            if not any(['reference' in text_lower, 'bibliography' in text_lower]):
                logger.error("No references or bibliography found")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating academic content: {str(e)}")
            return False

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
                async with aiofiles.open(full_path, 'w') as f:
                    await f.write(content)

                # Create file record
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
            logger.error(f"Error processing files: {str(e)}")
            raise FileHandlingError(
                f"File processing failed: {str(e)}",
                operation="process_files",
                file_type="programming"
            )

    async def _create_file_record(
        self,
        file_path: Path,
        file_type: str,
        original_name: str
    ) -> AssignmentFile:
        """
        Create a file record in the database
        
        Args:
            file_path: Path to the physical file
            file_type: Type of file (e.g., 'programming', 'academic_writing')
            original_name: Original name of the file
            
        Returns:
            AssignmentFile instance
            
        Raises:
            FileHandlingError: If file creation or database operation fails
        """
        try:
            # Read file content
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
                
            try:
                # Create Django file object from content
                file_obj = ContentFile(
                    content,
                    name=original_name
                )
                
                # Create AssignmentFile record
                assignment_file = await sync_to_async(AssignmentFile.objects.create)(
                    file=file_obj,
                    origin=f"{file_type}_submission"
                )
                
                # Update file path to be relative to MEDIA_ROOT
                assignment_file.file.name = str(file_path.relative_to(settings.MEDIA_ROOT))
                await sync_to_async(assignment_file.save)()
                
                return assignment_file

            except Exception as db_error:
                logger.error(f"Database error creating file record: {str(db_error)}")
                raise FileHandlingError(
                    "Failed to create database record",
                    operation="create_record",
                    file_path=str(file_path),
                    file_type=file_type
                ) from db_error

        except aiofiles.errors.AIOFilesError as file_error:
            logger.error(f"File reading error: {str(file_error)}")
            raise FileHandlingError(
                "Failed to read file",
                operation="read",
                file_path=str(file_path),
                file_type=file_type
            ) from file_error
            
        except Exception as e:
            logger.error(f"Unexpected error creating file record: {str(e)}")
            raise FileHandlingError(
                "Unexpected error creating file record",
                operation="create_record",
                file_path=str(file_path),
                file_type=file_type
            ) from e

    def _calculate_word_count(self, text: str) -> int:
        """Calculate target word count based on description"""
        words = len(text.split())
        # Use a simple multiplier based on description length
        return max(1000, min(5000, words * 3))

    async def initialize(self) -> None:
        """Initialize agent resources and connections"""
        try:
            logger.info("Initializing Planning Agent (Agent 3)")
            
            # Ensure submission directory exists
            self.submission_base_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize LLM
            self.model = ChatOpenAI(model="gpt-4")
            self.parser = JsonOutputParser()
            
            # Test Agent 4 connection
            await self._test_agent4_connection()
            
            # Test Agent 5 connection
            await self._test_agent5_connection()
            
            logger.info("Planning Agent initialization completed successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize Planning Agent: {str(e)}"
            logger.error(error_msg)
            raise AgentInitializationError(
                error_msg,
                reason="initialization_failed",
                details={"error": str(e)}
            )

    async def shutdown(self) -> None:
        """Cleanup resources during shutdown"""
        try:
            logger.info("Shutting down Planning Agent")
            
            # Cleanup any temporary files
            temp_files = list(self.submission_base_path.glob("**/temp_*"))
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")
            
            logger.info("Planning Agent shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during Planning Agent shutdown: {str(e)}")

    async def _test_agent4_connection(self) -> None:
        """Test connection to Agent 4"""
        try:
            test_request = ServiceRequest(
                request_id="test",
                message_type=MessageType.SERVICE_SPEC,
                service_spec={"type": "connection_test"}
            )
            await handle_programming_request(test_request)
        except Exception as e:
            raise AgentInitializationError(
                "Failed to connect to Agent 4",
                reason="agent4_connection_failed",
                details={"error": str(e)}
            )

    async def _test_agent5_connection(self) -> None:
        """Test connection to Agent 5"""
        try:
            # Import needed only for testing
            from .main_agent_five import academic_writing_pipeline
            if not callable(academic_writing_pipeline):
                raise Exception("Academic writing pipeline not properly configured")
        except Exception as e:
            raise AgentInitializationError(
                "Failed to connect to Agent 5",
                reason="agent5_connection_failed",
                details={"error": str(e)}
            )
    
    async def _determine_base_image(self, description: str) -> Dict[str, Any]:
        """
        Analyze project description to determine appropriate Docker base image.
        Uses LLM to understand requirements and suggest appropriate configuration.
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """As a Docker expert, analyze this project description and determine the base image requirements.
                Consider:
                1. Programming language and runtime environment
                2. Required system dependencies
                3. Required build tools
                4. Security considerations
                5. Any specialized requirements
                
                Return a JSON with:
                {
                    "base_image": {
                        "name": "base image to use",
                        "tag": "specific version tag"
                    },
                    "system_dependencies": ["required packages"],
                    "build_stage_dependencies": ["build-time dependencies"],
                    "runtime_dependencies": ["runtime dependencies"],
                    "build_args": {
                        "arg_name": "arg_value"
                    }
                }"""),
                ("human", "Project Description: {description}")
            ])

            result = await (prompt | self.model | self.parser).ainvoke({
                "description": description
            })

            # Validate result structure
            required_fields = ["base_image", "system_dependencies", "build_stage_dependencies"]
            if not all(field in result for field in required_fields):
                raise ValidationError(
                    "Incomplete Docker configuration",
                    validation_type="docker",
                    field="base_image",
                    details={"missing_fields": [f for f in required_fields if f not in result]}
                )

            return result

        except Exception as e:
            logger.error(f"Error determining base image: {str(e)}")
            raise

    async def _extract_env_requirements(
        self,
        execution_plan: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Extract and validate environment variables from execution plan.
        """
        try:
            env_vars = {}
            
            # Extract explicit environment variables from steps
            for step in execution_plan.get("steps", []):
                if "environment" in step:
                    env_vars.update(step["environment"])

            # Extract environment requirements from service configurations
            if "environment_requirements" in execution_plan:
                env_vars.update(execution_plan["environment_requirements"])

            return env_vars

        except Exception as e:
            logger.error(f"Error extracting environment requirements: {str(e)}")
            raise

    async def _determine_network_requirements(
        self,
        execution_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze execution plan to determine network requirements.
        Handles service discovery, port mappings, and network configurations.
        """
        try:
            network_config = {
                "exposed_ports": [],
                "internal_ports": [],
                "networks": [],
                "network_mode": "bridge",
                "port_mappings": {}
            }

            # Analyze services for networking needs
            for step in execution_plan.get("steps", []):
                if "service" in step:
                    service = step["service"]
                    if "networking" in service:
                        network_config = self._process_service_networking(
                            service["networking"],
                            network_config
                        )

            return network_config

        except Exception as e:
            logger.error(f"Error determining network requirements: {str(e)}")
            raise

    def _process_service_networking(
        self,
        service_networking: Dict[str, Any],
        network_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process networking requirements for a specific service"""
        if "ports" in service_networking:
            for port_config in service_networking["ports"]:
                port_info = {
                    "port": port_config["port"],
                    "protocol": port_config.get("protocol", "tcp"),
                    "purpose": port_config.get("purpose", "service"),
                    "is_required": port_config.get("required", True)
                }
                network_config["exposed_ports"].append(port_info)

        if "internal_ports" in service_networking:
            network_config["internal_ports"].extend(
                service_networking["internal_ports"]
            )

        return network_config

    def _process_project_networking(
        self,
        project_networking: Dict[str, Any],
        network_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process project-level networking requirements"""
        if "required_networks" in project_networking:
            network_config["required_networks"].extend(
                project_networking["required_networks"]
            )

        if "network_mode" in project_networking:
            network_config["network_mode"] = project_networking["network_mode"]

        return network_config
    
    async def _handle_container_operations(
        self,
        assignment: Assignment,
        generated_files: Dict[str, str],
        execution_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle Docker container operations for programming assignments.
        Creates and manages containers based on execution requirements.
        """
        try:
            # Get Docker configuration
            docker_config = {
                "base_image": await self._determine_base_image(assignment.description),
                "env_vars": await self._extract_env_requirements(execution_plan),
                "network": await self._determine_network_requirements(execution_plan)
            }

            # Build container using DockerClientManager
            image_info = await self.docker_manager.build_image(
                context_path=str(self.submission_base_path / f"assignment_{assignment.id}"),
                tag=f"assignment_{assignment.id}:latest",
                build_args=docker_config["base_image"].get("build_args", {})
            )

            # Configure container
            container_info = await self.docker_manager.create_container(
                image_id=image_info['image_id'],
                container_name=f"assignment_{assignment.id}",
                env_vars=docker_config["env_vars"],
                network_config=docker_config["network"]
            )

            return {
                "image": image_info,
                "container": container_info,
                "config": docker_config
            }

        except Exception as e:
            logger.error(f"Error in container operations: {str(e)}")
            raise

    async def _validate_container(
        self,
        container_id: str,
        validation_criteria: Dict[str, Any]
    ) -> bool:
        """
        Validate container meets requirements
        """
        try:
            # Get container status
            container_status = await self.docker_manager.get_container_status(container_id)
            
            # Check if container is running
            if container_status.get('State', {}).get('Status') != 'running':
                logger.error(f"Container {container_id} is not running")
                return False

            # Check port mappings
            port_mappings = container_status.get('NetworkSettings', {}).get('Ports', {})
            if not port_mappings:
                logger.error(f"No port mappings found for container {container_id}")
                return False

            # Validate exposed services are accessible
            for port_info in validation_criteria.get('required_ports', []):
                if not await self._check_port_accessibility(
                    container_id,
                    port_info['port'],
                    port_info.get('protocol', 'tcp')
                ):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating container: {str(e)}")
            return False

    async def _check_port_accessibility(
        self,
        container_id: str,
        port: int,
        protocol: str = 'tcp'
    ) -> bool:
        """
        Check if a port is accessible on the container
        """
        try:
            port_info = await self.docker_manager.check_port(
                container_id,
                port,
                protocol
            )
            return port_info.get('accessible', False)

        except Exception as e:
            logger.error(f"Error checking port accessibility: {str(e)}")
            return False

    async def _cleanup_containers(
        self,
        assignment_id: int,
        container_ids: List[str]
    ) -> None:
        """
        Clean up containers after assignment completion or failure
        """
        try:
            for container_id in container_ids:
                try:
                    await self.docker_manager.remove_container(
                        container_id,
                        force=True
                    )
                except Exception as container_e:
                    logger.error(
                        f"Error removing container {container_id}: {str(container_e)}"
                    )

        except Exception as e:
            logger.error(f"Error in container cleanup: {str(e)}")

# Global instance of the Planning Agent
_planning_agent: Optional[PlanningAgent] = None

async def get_planning_agent() -> PlanningAgent:
    """Get or create the global Planning Agent instance"""
    global _planning_agent
    
    if _planning_agent is None:
        _planning_agent = PlanningAgent()
        await _planning_agent.initialize()
    
    return _planning_agent

async def handle_assignment(assignment_id: int) -> Dict[str, Any]:
    """
    Main entry point for handling assignments.
    
    Args:
        assignment_id: ID of the Assignment record in the database
        
    Returns:
        Dict containing:
        - status: Success/failure status
        - assignment_id: The processed assignment ID
        - execution_plan: The plan that was executed
        - result: Results including any generated files or submissions
        
    Raises:
        ValidationError: If assignment validation fails
        ExecutionError: If execution fails
        FileHandlingError: If file processing fails
        PlanningError: If plan creation fails
        AgentInitializationError: If agent initialization fails
    """
    try:
        # Get the agent instance
        agent = await get_planning_agent()
        
        # Process the assignment
        logger.info(f"Starting assignment processing for ID: {assignment_id}")
        result = await agent.handle_assignment(assignment_id)
        logger.info(f"Successfully completed assignment {assignment_id}")
        
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error for assignment {assignment_id}: {str(e)}")
        raise
    except ExecutionError as e:
        logger.error(f"Execution error for assignment {assignment_id}: {str(e)}")
        raise
    except FileHandlingError as e:
        logger.error(f"File handling error for assignment {assignment_id}: {str(e)}")
        raise
    except PlanningError as e:
        logger.error(f"Planning error for assignment {assignment_id}: {str(e)}")
        raise
    except AgentInitializationError as e:
        logger.error(f"Agent initialization error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing assignment {assignment_id}: {str(e)}")
        raise

async def shutdown_planning_agent() -> None:
    """Shutdown the planning agent and cleanup resources"""
    global _planning_agent
    
    if _planning_agent is not None:
        await _planning_agent.shutdown()
        _planning_agent = None
        logger.info("Planning Agent shut down successfully")
    
    
