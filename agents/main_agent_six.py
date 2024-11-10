import logging  # Used for logger
import json    # Used for JSON handling
from typing import Dict, List, Any, Optional  # Used for type hints
from django.utils import timezone  # Used in validation (upcoming methods)
from django.db import transaction  # Used in updates (upcoming methods)
from asgiref.sync import sync_to_async  # Used for async DB operations
from langchain_openai import ChatOpenAI  # Used for LLM
from langchain_core.prompts import ChatPromptTemplate  # Used for prompts
from langchain_core.output_parsers import JsonOutputParser  # Used in init


from assignments.models import (  # Used for DB models
    Assignment,
    AssignmentSubmission,
    AssignmentFile,
    RevisionFile
)
from containers.utils import DockerClientManager  # Used for Docker operations

logger = logging.getLogger(__name__)

class ReviewError(Exception):
    """Custom exception for review process errors"""
    def __init__(
        self,
        message: str,
        review_type: str = None,
        criteria: str = None,
        details: Dict = None
    ):
        self.review_type = review_type
        self.criteria = criteria
        self.details = details or {}
        self.message = self._format_message(message)
        super().__init__(self.message)

    def _format_message(self, message: str) -> str:
        parts = [message]
        if self.review_type:
            parts.append(f"Review Type: {self.review_type}")
        if self.criteria:
            parts.append(f"Failed Criteria: {self.criteria}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)

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

class QualityCheckError(Exception):
    """Custom exception for quality check failures"""
    def __init__(
        self,
        message: str,
        check_type: str = None,
        threshold: Any = None,
        actual_value: Any = None,
        details: Dict = None
    ):
        self.check_type = check_type
        self.threshold = threshold
        self.actual_value = actual_value
        self.details = details or {}
        self.message = self._format_message(message)
        super().__init__(self.message)

    def _format_message(self, message: str) -> str:
        parts = [message]
        if self.check_type:
            parts.append(f"Check Type: {self.check_type}")
        if self.threshold is not None:
            parts.append(f"Required Threshold: {self.threshold}")
        if self.actual_value is not None:
            parts.append(f"Actual Value: {self.actual_value}")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " | ".join(parts)

class ReviewAgent:
    """
    Agent 6 - Final Review Agent
    Responsible for:
    1. Final review of completed assignments
    2. Quality assurance checks
    3. Determining if revisions are needed
    4. Validating delivery requirements
    """

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4")
        self.parser = JsonOutputParser()
        self.docker_manager = DockerClientManager()
        self.review_criteria = {
            "programming": {
                "code_quality": {
                    "min_score": 0.8,
                    "required_checks": [
                        "syntax",
                        "style",
                        "complexity",
                        "maintainability"
                    ]
                },
                "testing": {
                    "coverage_threshold": 0.7,
                    "required_test_types": [
                        "unit",
                        "integration"
                    ]
                },
                "documentation": {
                    "required_sections": [
                        "setup",
                        "usage",
                        "api",
                        "deployment"
                    ]
                },
                "security": {
                    "required_checks": [
                        "dependencies",
                        "vulnerabilities",
                        "code_analysis"
                    ]
                },
                "containerization": {
                    "required_checks": [
                        "dockerfile",
                        "compose",
                        "networking",
                        "security"
                    ]
                }
            },
            "academic": {
                "content": {
                    "min_score": 0.8,
                    "required_checks": [
                        "originality",
                        "coherence",
                        "argument_quality",
                        "research_depth"
                    ]
                },
                "structure": {
                    "required_sections": [
                        "introduction",
                        "methodology",
                        "results",
                        "conclusion"
                    ]
                },
                "references": {
                    "min_count": 5,
                    "quality_check": True,
                    "citation_style": True
                },
                "formatting": {
                    "required_checks": [
                        "style_guide",
                        "layout",
                        "citations",
                        "bibliography"
                    ]
                }
            }
        }

    async def review_assignment(
        self,
        assignment_id: int,
        submission_id: int
    ) -> Dict[str, Any]:
        """
        Main entry point for reviewing an assignment
        """
        try:
            # Get assignment and submission
            assignment = await self._get_assignment(assignment_id)
            submission = await self._get_submission(submission_id)

            # Validate basic requirements
            await self._validate_submission_status(assignment, submission)

            # Perform review based on assignment type
            if assignment.assignment_type == 'P':
                review_result = await self._review_programming_assignment(
                    assignment,
                    submission
                )
            else:
                review_result = await self._review_academic_assignment(
                    assignment,
                    submission
                )

            # Update assignment status based on review
            await self._update_assignment_status(
                assignment,
                submission,
                review_result
            )

            return review_result

        except (ReviewError, ValidationError, QualityCheckError) as e:
            logger.error(f"Review failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in review process: {str(e)}")
            raise ReviewError(
                f"Review process failed: {str(e)}",
                review_type="unknown",
                details={"error": str(e)}
            )
        
    async def _get_assignment(self, assignment_id: int) -> Assignment:
        """Retrieve and validate assignment"""
        try:
            assignment = await sync_to_async(Assignment.objects.select_related(
                'agent', 'original_platform', 'original_account'
            ).get)(pk=assignment_id)
            return assignment
        except Assignment.DoesNotExist:
            raise ValidationError(
                f"Assignment {assignment_id} not found",
                validation_type="existence",
                field="assignment_id"
            )

    async def _get_submission(self, submission_id: int) -> AssignmentSubmission:
        """Retrieve and validate submission"""
        try:
            submission = await sync_to_async(AssignmentSubmission.objects.select_related(
                'assignment'
            ).prefetch_related('files').get)(pk=submission_id)
            return submission
        except AssignmentSubmission.DoesNotExist:
            raise ValidationError(
                f"Submission {submission_id} not found",
                validation_type="existence",
                field="submission_id"
            )

    async def _validate_submission_status(
        self,
        assignment: Assignment,
        submission: AssignmentSubmission
    ) -> None:
        """Validate submission status and requirements"""
        try:
            # Check assignment status
            if not assignment.has_deposit_been_paid:
                raise ValidationError(
                    "Assignment deposit not paid",
                    validation_type="payment",
                    field="has_deposit_been_paid"
                )

            # Check submission files exist
            if not await sync_to_async(submission.files.exists)():
                raise ValidationError(
                    "No files found in submission",
                    validation_type="files",
                    field="submission_files"
                )

            # Check time requirements
            current_time = timezone.now()
            expected_delivery = assignment.expected_delivery_time

            if current_time < expected_delivery:
                raise ValidationError(
                    "Cannot review before 60% of deadline",
                    validation_type="timing",
                    field="delivery_time",
                    details={
                        "current_time": current_time.isoformat(),
                        "expected_delivery": expected_delivery.isoformat()
                    }
                )

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Submission validation failed: {str(e)}",
                validation_type="general"
            )

    async def _review_programming_assignment(
        self,
        assignment: Assignment,
        submission: AssignmentSubmission
    ) -> Dict[str, Any]:
        """Review programming assignment submission"""
        try:
            # Get review criteria
            criteria = self.review_criteria["programming"]
            
            # Initialize review results
            review_results = {
                "passed": True,
                "reviews": {},
                "issues": [],
                "suggestions": []
            }

            # Review code quality
            code_review = await self._review_code_quality(
                submission,
                criteria["code_quality"]
            )
            review_results["reviews"]["code_quality"] = code_review
            if not code_review["passed"]:
                review_results["passed"] = False
                review_results["issues"].extend(code_review["issues"])

            # Review testing
            test_review = await self._review_testing(
                submission,
                criteria["testing"]
            )
            review_results["reviews"]["testing"] = test_review
            if not test_review["passed"]:
                review_results["passed"] = False
                review_results["issues"].extend(test_review["issues"])

            # Review documentation
            doc_review = await self._review_documentation(
                submission,
                criteria["documentation"]
            )
            review_results["reviews"]["documentation"] = doc_review
            if not doc_review["passed"]:
                review_results["passed"] = False
                review_results["issues"].extend(doc_review["issues"])

            # Review security
            security_review = await self._review_security(
                submission,
                criteria["security"]
            )
            review_results["reviews"]["security"] = security_review
            if not security_review["passed"]:
                review_results["passed"] = False
                review_results["issues"].extend(security_review["issues"])

            # Review containerization
            container_review = await self._review_containerization(
                submission,
                criteria["containerization"]
            )
            review_results["reviews"]["containerization"] = container_review
            if not container_review["passed"]:
                review_results["passed"] = False
                review_results["issues"].extend(container_review["issues"])

            return review_results

        except Exception as e:
            logger.error(f"Error reviewing programming assignment: {str(e)}")
            raise ReviewError(
                f"Programming review failed: {str(e)}",
                review_type="programming"
            )

    async def _review_academic_assignment(
        self,
        assignment: Assignment,
        submission: AssignmentSubmission
    ) -> Dict[str, Any]:
        """Review academic writing assignment submission"""
        try:
            # Get review criteria
            criteria = self.review_criteria["academic"]
            
            # Initialize review results
            review_results = {
                "passed": True,
                "reviews": {},
                "issues": [],
                "suggestions": []
            }

            # Review content quality
            content_review = await self._review_content_quality(
                submission,
                criteria["content"]
            )
            review_results["reviews"]["content"] = content_review
            if not content_review["passed"]:
                review_results["passed"] = False
                review_results["issues"].extend(content_review["issues"])

            # Review structure
            structure_review = await self._review_structure(
                submission,
                criteria["structure"]
            )
            review_results["reviews"]["structure"] = structure_review
            if not structure_review["passed"]:
                review_results["passed"] = False
                review_results["issues"].extend(structure_review["issues"])

            # Review references
            ref_review = await self._review_references(
                submission,
                criteria["references"]
            )
            review_results["reviews"]["references"] = ref_review
            if not ref_review["passed"]:
                review_results["passed"] = False
                review_results["issues"].extend(ref_review["issues"])

            # Review formatting
            format_review = await self._review_formatting(
                submission,
                criteria["formatting"]
            )
            review_results["reviews"]["formatting"] = format_review
            if not format_review["passed"]:
                review_results["passed"] = False
                review_results["issues"].extend(format_review["issues"])

            return review_results

        except Exception as e:
            logger.error(f"Error reviewing academic assignment: {str(e)}")
            raise ReviewError(
                f"Academic review failed: {str(e)}",
                review_type="academic"
            )
    
    async def _review_code_quality(
        self,
        submission: AssignmentSubmission,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review code quality using LLM and static analysis"""
        try:
            review_result = {
                "passed": True,
                "score": 0.0,
                "issues": [],
                "suggestions": [],
                "details": {}
            }

            # Prepare code content for review
            code_files = await self._get_code_files(submission)
            
            # LLM Analysis
            prompt = ChatPromptTemplate.from_messages([
                ("system", """As a code reviewer, analyze the provided code for:
                1. Code style and consistency
                2. Code complexity and maintainability
                3. Best practices adherence
                4. Potential bugs or issues
                5. Performance considerations
                
                Return a JSON with:
                {
                    "score": float,
                    "issues": [{"file": str, "line": int, "description": str, "severity": str}],
                    "suggestions": [{"description": str, "priority": str}],
                    "metrics": {
                        "complexity": float,
                        "maintainability": float,
                        "readability": float
                    }
                }"""),
                ("human", "Review this code:\n{code_content}")
            ])

            analysis = await (prompt | self.model | self.parser).ainvoke({
                "code_content": "\n=== File Separator ===\n".join(
                    [f"{path}:\n{content}" for path, content in code_files.items()]
                )
            })

            # Update review result
            review_result["score"] = analysis["score"]
            review_result["details"]["metrics"] = analysis["metrics"]
            
            # Check if meets minimum score
            if analysis["score"] < criteria["min_score"]:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "score",
                    "message": f"Code quality score {analysis['score']} below required {criteria['min_score']}"
                })

            # Add issues and suggestions
            review_result["issues"].extend(analysis["issues"])
            review_result["suggestions"].extend(analysis["suggestions"])

            return review_result

        except Exception as e:
            logger.error(f"Error in code quality review: {str(e)}")
            raise ReviewError(
                f"Code quality review failed: {str(e)}",
                review_type="code_quality"
            )

    async def _review_testing(
        self,
        submission: AssignmentSubmission,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review test coverage and quality"""
        try:
            review_result = {
                "passed": True,
                "coverage": 0.0,
                "issues": [],
                "suggestions": [],
                "details": {}
            }

            # Get test files and container info
            test_files = await self._get_test_files(submission)
            container_info = await self._get_container_info(submission)

            if not test_files:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "missing_tests",
                    "message": "No test files found in submission"
                })
                return review_result

            # Run tests in container
            test_results = await self._run_tests_in_container(
                container_info["container_id"],
                test_files
            )

            # Update review result
            review_result["coverage"] = test_results["coverage"]
            review_result["details"]["test_results"] = test_results["results"]

            # Check coverage threshold
            if test_results["coverage"] < criteria["coverage_threshold"]:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "low_coverage",
                    "message": f"Test coverage {test_results['coverage']}% below required {criteria['coverage_threshold']}%"
                })

            # Check required test types
            missing_types = [
                test_type for test_type in criteria["required_test_types"]
                if not test_results.get(f"{test_type}_tests")
            ]
            if missing_types:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "missing_test_types",
                    "message": f"Missing required test types: {', '.join(missing_types)}"
                })

            return review_result

        except Exception as e:
            logger.error(f"Error in testing review: {str(e)}")
            raise ReviewError(
                f"Testing review failed: {str(e)}",
                review_type="testing"
            )

    async def _review_security(
        self,
        submission: AssignmentSubmission,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review code security"""
        try:
            review_result = {
                "passed": True,
                "issues": [],
                "suggestions": [],
                "details": {}
            }

            # Get all submission files
            code_files = await self._get_code_files(submission)
            
            # Security analysis using LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", """As a security expert, analyze the code for:
                1. Security vulnerabilities
                2. Dependency issues
                3. Code security best practices
                4. Potential security risks
                5. Security configuration issues
                
                Return a JSON with:
                {
                    "vulnerabilities": [{
                        "type": str,
                        "severity": str,
                        "description": str,
                        "file": str,
                        "line": int
                    }],
                    "dependency_issues": [{
                        "package": str,
                        "issue": str,
                        "severity": str
                    }],
                    "configuration_issues": [{
                        "type": str,
                        "description": str,
                        "impact": str
                    }]
                }"""),
                ("human", "Analyze this code for security issues:\n{code_content}")
            ])

            analysis = await (prompt | self.model | self.parser).ainvoke({
                "code_content": "\n=== File Separator ===\n".join(
                    [f"{path}:\n{content}" for path, content in code_files.items()]
                )
            })

            # Process vulnerabilities
            critical_vulnerabilities = [
                v for v in analysis["vulnerabilities"]
                if v["severity"].lower() == "critical"
            ]
            if critical_vulnerabilities:
                review_result["passed"] = False
                review_result["issues"].extend([
                    {
                        "type": "security_vulnerability",
                        "message": v["description"],
                        "location": f"{v['file']}:{v['line']}"
                    }
                    for v in critical_vulnerabilities
                ])

            # Process dependency issues
            critical_dependencies = [
                d for d in analysis["dependency_issues"]
                if d["severity"].lower() == "critical"
            ]
            if critical_dependencies:
                review_result["passed"] = False
                review_result["issues"].extend([
                    {
                        "type": "dependency_issue",
                        "message": f"{d['package']}: {d['issue']}"
                    }
                    for d in critical_dependencies
                ])

            # Add all issues to details
            review_result["details"] = analysis

            return review_result

        except Exception as e:
            logger.error(f"Error in security review: {str(e)}")
            raise ReviewError(
                f"Security review failed: {str(e)}",
                review_type="security"
            )
    
    async def _review_documentation(
        self,
        submission: AssignmentSubmission,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review project documentation"""
        try:
            review_result = {
                "passed": True,
                "issues": [],
                "suggestions": [],
                "details": {}
            }

            # Get documentation files
            docs = await self._get_documentation_files(submission)
            if not docs:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "missing_documentation",
                    "message": "No documentation files found"
                })
                return review_result

            # Review documentation using LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", """As a technical documentation expert, analyze the provided documentation for:
                1. Completeness of required sections
                2. Clarity and understandability
                3. Technical accuracy
                4. Usage examples and instructions
                5. Setup and deployment guides
                
                Consider these required sections:
                {required_sections}
                
                Return a JSON with:
                {
                    "missing_sections": [str],
                    "incomplete_sections": [{
                        "section": str,
                        "issues": [str]
                    }],
                    "clarity_score": float,
                    "accuracy_score": float,
                    "suggestions": [{
                        "section": str,
                        "suggestion": str,
                        "priority": str
                    }]
                }"""),
                ("human", "Review this documentation:\n{documentation}")
            ])

            analysis = await (prompt | self.model | self.parser).ainvoke({
                "documentation": "\n=== Document Separator ===\n".join(
                    [f"{path}:\n{content}" for path, content in docs.items()]
                ),
                "required_sections": json.dumps(criteria["required_sections"])
            })

            # Check for missing required sections
            if analysis["missing_sections"]:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "missing_sections",
                    "message": f"Missing required sections: {', '.join(analysis['missing_sections'])}"
                })

            # Check for incomplete sections
            if analysis["incomplete_sections"]:
                review_result["issues"].extend([
                    {
                        "type": "incomplete_section",
                        "message": f"Section '{section['section']}' is incomplete: {', '.join(section['issues'])}"
                    }
                    for section in analysis["incomplete_sections"]
                ])

            # Check clarity and accuracy scores
            if analysis["clarity_score"] < 0.7:  # threshold for clarity
                review_result["issues"].append({
                    "type": "low_clarity",
                    "message": f"Documentation clarity score ({analysis['clarity_score']}) is below acceptable threshold"
                })

            if analysis["accuracy_score"] < 0.8:  # threshold for technical accuracy
                review_result["issues"].append({
                    "type": "low_accuracy",
                    "message": f"Documentation technical accuracy score ({analysis['accuracy_score']}) is below acceptable threshold"
                })

            # Add suggestions for improvement
            review_result["suggestions"].extend([
                {
                    "section": suggestion["section"],
                    "message": suggestion["suggestion"],
                    "priority": suggestion["priority"]
                }
                for suggestion in analysis["suggestions"]
            ])

            # Update passed status based on issues
            if review_result["issues"]:
                review_result["passed"] = False

            review_result["details"] = analysis
            return review_result

        except Exception as e:
            logger.error(f"Error in documentation review: {str(e)}")
            raise ReviewError(
                f"Documentation review failed: {str(e)}",
                review_type="documentation"
            )

    async def _review_containerization(
        self,
        submission: AssignmentSubmission,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review Docker configuration and container setup"""
        try:
            review_result = {
                "passed": True,
                "issues": [],
                "suggestions": [],
                "details": {}
            }

            # Get Docker-related files
            docker_files = await self._get_docker_files(submission)
            if not docker_files:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "missing_docker_files",
                    "message": "No Docker configuration files found"
                })
                return review_result

            # Check required files
            required_files = ['Dockerfile', 'docker-compose.yml']
            missing_files = [
                file for file in required_files
                if file not in docker_files
            ]
            if missing_files:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "missing_required_files",
                    "message": f"Missing required Docker files: {', '.join(missing_files)}"
                })

            # Review Docker configuration using LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", """As a Docker expert, analyze the Docker configuration for:
                1. Dockerfile best practices
                2. Security considerations
                3. Multi-stage builds efficiency
                4. Network configuration
                5. Volume management
                6. Environment configuration
                
                Return a JSON with:
                {
                    "dockerfile_issues": [{
                        "line": int,
                        "issue": str,
                        "severity": str,
                        "suggestion": str
                    }],
                    "compose_issues": [{
                        "service": str,
                        "issue": str,
                        "severity": str,
                        "suggestion": str
                    }],
                    "security_issues": [{
                        "type": str,
                        "description": str,
                        "severity": str,
                        "mitigation": str
                    }],
                    "best_practices_score": float
                }"""),
                ("human", "Review these Docker configurations:\n{docker_files}")
            ])

            analysis = await (prompt | self.model | self.parser).ainvoke({
                "docker_files": "\n=== File Separator ===\n".join(
                    [f"{path}:\n{content}" for path, content in docker_files.items()]
                )
            })

            # Process Dockerfile issues
            critical_dockerfile_issues = [
                issue for issue in analysis["dockerfile_issues"]
                if issue["severity"].lower() == "critical"
            ]
            if critical_dockerfile_issues:
                review_result["passed"] = False
                review_result["issues"].extend([
                    {
                        "type": "dockerfile_issue",
                        "message": f"Line {issue['line']}: {issue['issue']}"
                    }
                    for issue in critical_dockerfile_issues
                ])

            # Process compose issues
            critical_compose_issues = [
                issue for issue in analysis["compose_issues"]
                if issue["severity"].lower() == "critical"
            ]
            if critical_compose_issues:
                review_result["passed"] = False
                review_result["issues"].extend([
                    {
                        "type": "compose_issue",
                        "message": f"Service {issue['service']}: {issue['issue']}"
                    }
                    for issue in critical_compose_issues
                ])

            # Process security issues
            critical_security_issues = [
                issue for issue in analysis["security_issues"]
                if issue["severity"].lower() == "critical"
            ]
            if critical_security_issues:
                review_result["passed"] = False
                review_result["issues"].extend([
                    {
                        "type": "docker_security",
                        "message": f"{issue['type']}: {issue['description']}"
                    }
                    for issue in critical_security_issues
                ])

            # Check best practices score
            if analysis["best_practices_score"] < 0.8:  # threshold for best practices
                review_result["issues"].append({
                    "type": "best_practices",
                    "message": f"Docker configuration best practices score ({analysis['best_practices_score']}) is below acceptable threshold"
                })

            # Add all issues to details
            review_result["details"] = analysis

            # Test container build and run
            try:
                container_test = await self._test_container_build(submission, docker_files)
                review_result["details"]["container_test"] = container_test
                if not container_test["success"]:
                    review_result["passed"] = False
                    review_result["issues"].extend(container_test["issues"])
            except Exception as container_e:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "container_test_failed",
                    "message": str(container_e)
                })

            return review_result

        except Exception as e:
            logger.error(f"Error in containerization review: {str(e)}")
            raise ReviewError(
                f"Containerization review failed: {str(e)}",
                review_type="containerization"
            )
    
    async def _get_code_files(
            self,
            submission: AssignmentSubmission
        ) -> Dict[str, str]:
            """Get code files from submission"""
            try:
                code_files = {}
                async for file in submission.files.aiterator():
                    if file.file.name.endswith(('.py', '.js', '.java', '.cpp', '.html', '.css')):
                        content = await sync_to_async(self._read_file)(file.file.path)
                        code_files[file.file.name] = content
                return code_files
            except Exception as e:
                logger.error(f"Error getting code files: {str(e)}")
                raise ReviewError("Failed to read code files")

    def _read_file(self, path: str) -> str:
        """Synchronous file reading"""
        with open(path, 'r') as f:
            return f.read()

    async def _get_documentation_files(
            self,
            submission: AssignmentSubmission
        ) -> Dict[str, str]:
            """Get documentation files from submission"""
            try:
                doc_files = {}
                async for file in submission.files.aiterator():
                    if file.file.name.endswith(('.md', '.txt', '.rst', '.doc', '.docx')):
                        content = await sync_to_async(self._read_file)(file.file.path)
                        doc_files[file.file.name] = content
                return doc_files
            except Exception as e:
                logger.error(f"Error getting documentation files: {str(e)}")
                raise ReviewError("Failed to read documentation files")

    async def _get_docker_files(
        self,
        submission: AssignmentSubmission
    ) -> Dict[str, str]:
        """Get Docker-related files from submission"""
        try:
            docker_files = {}
            async for file in submission.files.aiterator():
                if file.file.name in ['Dockerfile', 'docker-compose.yml']:
                    content = await sync_to_async(self._read_file)(file.file.path)
                    docker_files[file.file.name] = content
            return docker_files
        except Exception as e:
            logger.error(f"Error getting Docker files: {str(e)}")
            raise ReviewError("Failed to read Docker files")

    def _read_file(self, path: str) -> str:
        """Synchronous file reading"""
        with open(path, 'r') as f:
            return f.read()

    async def _test_container_build(
        self,
        submission: AssignmentSubmission,
        docker_files: Dict[str, str]
    ) -> Dict[str, Any]:
        """Test if container builds and runs"""
        try:
            # Get container info from submission
            container_info = submission.get_container_info()
            if not container_info:
                return {
                    "success": False,
                    "issues": [{
                        "type": "container_missing",
                        "message": "No container information found"
                    }]
                }

            # Test container
            container_status = await self.docker_manager.get_container_status(
                container_info['container_id']
            )

            return {
                "success": container_status.get('State', {}).get('Status') == 'running',
                "container_status": container_status
            }

        except Exception as e:
            logger.error(f"Error testing container: {str(e)}")
            return {
                "success": False,
                "issues": [{
                    "type": "container_test_failed",
                    "message": str(e)
                }]
            }
        
    async def _review_content_quality(
        self,
        submission: AssignmentSubmission,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review academic content quality"""
        try:
            review_result = {
                "passed": True,
                "issues": [],
                "suggestions": []
            }

            # Get document content
            content = await self._get_document_content(submission)
            if not content:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "missing_content",
                    "message": "No content found in submission"
                })
                return review_result

            # Review content using LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Review this academic content for:
                1. Argument quality and coherence
                2. Research depth
                3. Originality
                4. Academic standards
                
                Return a JSON with:
                {
                    "score": float,
                    "issues": [{"issue": str, "severity": str}],
                    "suggestions": [{"suggestion": str}]
                }"""),
                ("human", "Review this content:\n{content}")
            ])

            analysis = await (prompt | self.model | self.parser).ainvoke({
                "content": content
            })

            # Check if meets minimum score
            if analysis["score"] < criteria["min_score"]:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "quality_score",
                    "message": f"Content quality score {analysis['score']} below required {criteria['min_score']}"
                })

            # Add critical issues
            critical_issues = [
                issue for issue in analysis["issues"]
                if issue["severity"] == "critical"
            ]
            if critical_issues:
                review_result["passed"] = False
                review_result["issues"].extend([
                    {"type": "content", "message": issue["issue"]}
                    for issue in critical_issues
                ])

            review_result["suggestions"].extend(analysis["suggestions"])
            return review_result

        except Exception as e:
            logger.error(f"Error in content quality review: {str(e)}")
            raise ReviewError(f"Content quality review failed: {str(e)}")

    async def _review_structure(
        self,
        submission: AssignmentSubmission,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review academic document structure"""
        try:
            review_result = {
                "passed": True,
                "issues": [],
                "suggestions": []
            }

            content = await self._get_document_content(submission)
            if not content:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "missing_content",
                    "message": "No content found in submission"
                })
                return review_result

            # Check required sections
            missing_sections = []
            for section in criteria["required_sections"]:
                # Simple check for section headers
                if not any(
                    line.lower().strip().startswith(section.lower())
                    for line in content.split('\n')
                ):
                    missing_sections.append(section)

            if missing_sections:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "missing_sections",
                    "message": f"Missing required sections: {', '.join(missing_sections)}"
                })

            return review_result

        except Exception as e:
            logger.error(f"Error in structure review: {str(e)}")
            raise ReviewError(f"Structure review failed: {str(e)}")

    async def _review_references(
        self,
        submission: AssignmentSubmission,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review academic references and citations"""
        try:
            review_result = {
                "passed": True,
                "issues": [],
                "suggestions": []
            }

            content = await self._get_document_content(submission)
            if not content:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "missing_content",
                    "message": "No content found in submission"
                })
                return review_result

            # Review references using LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Review the academic references for:
                1. Number of references
                2. Citation quality
                3. Reference formatting
                
                Return a JSON with:
                {
                    "reference_count": int,
                    "citation_issues": [{"issue": str, "severity": str}],
                    "format_issues": [{"issue": str}]
                }"""),
                ("human", "Review these references:\n{content}")
            ])

            analysis = await (prompt | self.model | self.parser).ainvoke({
                "content": content
            })

            # Check minimum reference count
            if analysis["reference_count"] < criteria["min_count"]:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "reference_count",
                    "message": f"Found {analysis['reference_count']} references, minimum required is {criteria['min_count']}"
                })

            # Check citation issues
            critical_citation_issues = [
                issue for issue in analysis["citation_issues"]
                if issue["severity"] == "critical"
            ]
            if critical_citation_issues:
                review_result["passed"] = False
                review_result["issues"].extend([
                    {"type": "citation", "message": issue["issue"]}
                    for issue in critical_citation_issues
                ])

            # Add formatting issues as suggestions
            review_result["suggestions"].extend([
                {"message": issue["issue"]}
                for issue in analysis["format_issues"]
            ])

            return review_result

        except Exception as e:
            logger.error(f"Error in references review: {str(e)}")
            raise ReviewError(f"References review failed: {str(e)}")

    async def _review_formatting(
        self,
        submission: AssignmentSubmission,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review academic document formatting"""
        try:
            review_result = {
                "passed": True,
                "issues": [],
                "suggestions": []
            }

            content = await self._get_document_content(submission)
            if not content:
                review_result["passed"] = False
                review_result["issues"].append({
                    "type": "missing_content",
                    "message": "No content found in submission"
                })
                return review_result

            # Review formatting using LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Review the document formatting for:
                1. Academic style guide compliance
                2. Layout consistency
                3. Citation formatting
                4. Overall presentation
                
                Return a JSON with:
                {
                    "format_issues": [{"issue": str, "severity": str}],
                    "style_guide_issues": [{"issue": str, "severity": str}]
                }"""),
                ("human", "Review this document formatting:\n{content}")
            ])

            analysis = await (prompt | self.model | self.parser).ainvoke({
                "content": content
            })

            # Process critical formatting issues
            critical_issues = [
                issue for issue in analysis["format_issues"] + analysis["style_guide_issues"]
                if issue["severity"] == "critical"
            ]
            
            if critical_issues:
                review_result["passed"] = False
                review_result["issues"].extend([
                    {"type": "formatting", "message": issue["issue"]}
                    for issue in critical_issues
                ])

            return review_result

        except Exception as e:
            logger.error(f"Error in formatting review: {str(e)}")
            raise ReviewError(f"Formatting review failed: {str(e)}")

    async def _get_document_content(
        self,
        submission: AssignmentSubmission
    ) -> Optional[str]:
        """Get academic document content"""
        try:
            document_files = {}
            async for file in submission.files.aiterator():
                if file.file.name.endswith(('.doc', '.docx', '.pdf', '.txt')):
                    content = await sync_to_async(self._read_file)(file.file.path)
                    document_files[file.file.name] = content

            if not document_files:
                return None

            # Combine content from all files
            return "\n=== Document Separator ===\n".join(document_files.values())

        except Exception as e:
            logger.error(f"Error getting document content: {str(e)}")
            return None
        
    async def _update_assignment_status(
        self,
        assignment: Assignment,
        submission: AssignmentSubmission,
        review_result: Dict[str, Any]
    ) -> None:
        """Update assignment status based on review results"""
        try:
            with transaction.atomic():
                if review_result["passed"]:
                    # Passed review - mark ready for delivery
                    assignment.completed = True
                    assignment.has_revisions = False
                    submission.status = "approved"
                else:
                    # Failed review - needs revision
                    assignment.completed = False
                    assignment.has_revisions = True
                    submission.status = "needs_revision"
                    
                # Save changes
                await sync_to_async(assignment.save)()
                await sync_to_async(submission.save)()

                # Notify Agent 3 of results
                await self._notify_agent_three(
                    assignment,
                    review_result,
                    passed=review_result["passed"]
                )

        except Exception as e:
            logger.error(f"Error updating assignment status: {str(e)}")
            raise

    async def _notify_agent_three(
        self,
        assignment: Assignment,
        review_result: Dict[str, Any],
        passed: bool
    ) -> None:
        """
        Notify Agent 3 of review results.
        If review fails, assignment goes back to Agent 3 for correction.
        If review passes, assignment is marked ready for delivery.
        """
        try:
            if not passed:
                # Send back to Agent 3 with review feedback
                assignment.completed = False  # Reset completion status
                assignment.review_feedback = json.dumps(review_result)
                await sync_to_async(assignment.save)()

                # Send back to Agent 3 for correction
                from .main_agent_three import handle_assignment
                await handle_assignment(assignment.id)
            else:
                # Mark assignment as reviewed and ready for delivery
                assignment.review_passed = True
                assignment.review_date = timezone.now()
                await sync_to_async(assignment.save)()

                logger.info(
                    f"Assignment {assignment.id} passed final review - "
                    f"ready for delivery"
                )

        except Exception as e:
            logger.error(f"Error notifying Agent 3: {str(e)}")
            raise
