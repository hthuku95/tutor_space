
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

import asyncio
import json
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .protocols.agent_protocol import (
    AgentProtocol,
    ServiceRequest,
    ServiceResponse,
    CodeFile,
    MessageType
)
from containers.utils import DockerClientManager
from django.conf import settings

logger = logging.getLogger(__name__)

class ProgrammingAgent:
    """
    Agent 4 - Handles programming tasks and code generation based on service requirements.
    Capabilities:
    - Code generation
    - Infrastructure configuration
    - Validation and testing
    - Integration setup
    """

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4")
        self.protocol = AgentProtocol()
        self.docker_manager = DockerClientManager()
        self.request_queue = asyncio.Queue()
        self.ongoing_tasks: Dict[str, asyncio.Task] = {}

    async def start(self):
        """Start the agent's main processing loop"""
        logger.info("Starting Programming Agent (Agent 4)")
        try:
            while True:
                request = await self.request_queue.get()
                if self.protocol.validate_request(request):
                    task = asyncio.create_task(self._process_request(request))
                    self.ongoing_tasks[request.request_id] = task
                    await self._cleanup_completed_tasks()
                else:
                    logger.error(f"Invalid request received: {request}")

        except Exception as e:
            logger.error(f"Error in agent main loop: {str(e)}")
            raise

    async def handle_request(self, request: ServiceRequest) -> Dict[str, Any]:
        """Handle incoming requests from other agents"""
        try:
            logger.info(f"Received request: {request.request_id}")
            await self.request_queue.put(request)
            return {"status": "queued", "request_id": request.request_id}
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            raise

    async def _process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process a single service request"""
        try:
            logger.info(f"Processing request: {request.request_id}")
            response = None

            if request.message_type == MessageType.SERVICE_SPEC:
                response = await self._handle_code_generation(request)
            elif request.message_type == MessageType.CONFIG_REQUEST:
                response = await self._handle_validation(request)
            else:
                raise ValueError(f"Unsupported message type: {request.message_type}")

            return response

        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {str(e)}")
            return self._create_error_response(request, str(e))

    async def _handle_code_generation(self, request: ServiceRequest) -> ServiceResponse:
        """Handle code generation requests"""
        try:
            # Analyze requirements and plan implementation
            implementation_plan = await self._create_implementation_plan(request)

            # Generate code based on the plan
            generated_files = await self._generate_implementation(
                implementation_plan,
                request.service_spec
            )

            # Perform initial validation
            validation_results = await self._validate_implementation(
                generated_files,
                request.service_spec
            )

            return self.protocol.create_service_response(
                request_id=request.request_id,
                status="completed",
                files=[
                    CodeFile(
                        path=path,
                        content=content,
                        language=implementation_plan["file_specs"][path]["language"],
                        purpose=implementation_plan["file_specs"][path]["purpose"]
                    )
                    for path, content in generated_files.items()
                ],
                additional_info={
                    "validation_results": validation_results,
                    "implementation_plan": implementation_plan
                }
            )

        except Exception as e:
            logger.error(f"Error in code generation: {str(e)}")
            return self._create_error_response(request, str(e))

    async def _create_implementation_plan(
        self,
        request: ServiceRequest
    ) -> Dict[str, Any]:
        """Create a detailed implementation plan based on requirements"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the service requirements and create a detailed implementation plan.
            The plan should:
            1. Be technology and language agnostic
            2. Focus on fulfilling functional and non-functional requirements
            3. Define component relationships and interactions
            4. Specify testing and validation approaches
            5. Consider scalability and maintainability
            
            Do not make assumptions about specific:
            - Programming languages
            - Frameworks or libraries
            - Implementation patterns
            - Development tools
            
            Instead, derive everything from the requirements."""),
            ("human", """Create an implementation plan for:
            Service Specification: {service_spec}
            Requirements: {requirements}
            
            Return a structured implementation plan.""")
        ])

        chain = prompt | self.model | StrOutputParser()

        try:
            plan = await chain.ainvoke({
                "service_spec": request.service_spec,
                "requirements": request.requirements
            })
            
            return json.loads(plan)

        except Exception as e:
            logger.error(f"Error creating implementation plan: {str(e)}")
            raise

    async def _generate_implementation(
        self,
        implementation_plan: Dict[str, Any],
        service_spec: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate implementation based on the plan"""
        generated_files = {}
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Generate implementation code that:
                1. Fulfills all specified requirements
                2. Follows best practices for the chosen approach
                3. Includes proper error handling and logging
                4. Implements necessary tests
                5. Provides clear documentation
                
                The implementation should be:
                - Clean and maintainable
                - Properly structured
                - Well-documented
                - Fully tested
                
                Focus on requirements, not specific technologies."""),
                ("human", """Generate implementation for:
                Component: {component}
                Requirements: {requirements}
                Context: {context}
                
                Return the complete implementation.""")
            ])

            chain = prompt | self.model | StrOutputParser()

            for component in implementation_plan["components"]:
                content = await chain.ainvoke({
                    "component": component,
                    "requirements": service_spec["requirements"],
                    "context": {
                        "implementation_plan": implementation_plan,
                        "service_spec": service_spec
                    }
                })
                
                generated_files[component["path"]] = content

            return generated_files

        except Exception as e:
            logger.error(f"Error generating implementation: {str(e)}")
            raise

    async def _handle_validation(self, request: ServiceRequest) -> ServiceResponse:
        """Handle validation requests"""
        try:
            # Create validation plan
            validation_plan = await self._create_validation_plan(
                request.files,
                request.service_spec
            )

            # Execute validation
            validation_results = await self._execute_validation(
                validation_plan,
                request.files,
                request.service_spec
            )

            return self.protocol.create_service_response(
                request_id=request.request_id,
                status="valid" if validation_results["is_valid"] else "invalid",
                files=[],
                additional_info=validation_results
            )

        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            return self._create_error_response(request, str(e))

    async def _create_validation_plan(
        self,
        files: Dict[str, str],
        service_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a validation plan based on requirements"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Create a validation plan that verifies:
            1. Functional requirements
            2. Non-functional requirements
            3. Integration points
            4. Security requirements
            5. Performance requirements
            
            The plan should be:
            - Requirement-based
            - Technology-agnostic
            - Comprehensive
            - Measurable
            
            Focus on validation criteria, not specific tools."""),
            ("human", """Create validation plan for:
            Service Specification: {service_spec}
            Files: {files}
            
            Return a structured validation plan.""")
        ])

        chain = prompt | self.model | StrOutputParser()

        try:
            plan = await chain.ainvoke({
                "service_spec": service_spec,
                "files": list(files.keys())
            })
            
            return json.loads(plan)

        except Exception as e:
            logger.error(f"Error creating validation plan: {str(e)}")
            raise

    async def _execute_validation(
        self,
        validation_plan: Dict[str, Any],
        files: Dict[str, str],
        service_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute validation according to the plan"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Validate the implementation against requirements.
            Check for:
            1. Requirement fulfillment
            2. Implementation correctness
            3. Best practices adherence
            4. Security considerations
            5. Performance implications
            
            Provide detailed feedback on:
            - Issues found
            - Improvement suggestions
            - Compliance status"""),
            ("human", """Validate implementation:
            Validation Plan: {validation_plan}
            Files: {files}
            Requirements: {requirements}
            
            Return detailed validation results.""")
        ])

        chain = prompt | self.model | StrOutputParser()

        try:
            results = await chain.ainvoke({
                "validation_plan": validation_plan,
                "files": files,
                "requirements": service_spec["requirements"]
            })
            
            return json.loads(results)

        except Exception as e:
            logger.error(f"Error executing validation: {str(e)}")
            raise

    async def _cleanup_completed_tasks(self):
        """Clean up completed tasks"""
        completed = [
            req_id for req_id, task in self.ongoing_tasks.items()
            if task.done()
        ]
        for req_id in completed:
            del self.ongoing_tasks[req_id]

    def _create_error_response(
        self,
        request: ServiceRequest,
        error_message: str
    ) -> ServiceResponse:
        """Create an error response"""
        return self.protocol.create_service_response(
            request_id=request.request_id,
            status="failed",
            files=[],
            additional_info={"error": error_message}
        )

# Initialize the agent
programming_agent = ProgrammingAgent()

# Function to start the agent
async def start_agent():
    await programming_agent.start()

# Function to handle incoming requests
async def handle_request(request: ServiceRequest):
    return await programming_agent.handle_request(request)
