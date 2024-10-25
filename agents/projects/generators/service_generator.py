
import os
import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from ...types import (
    ProjectConfig,
    ServiceConfig,
    ServiceType,
    DeploymentType,
    ResourceRequirements
)
from ..protocols.agent_protocol import (
    AgentProtocol,
    CodeGenerationPriority,
    ServiceRequest,
    ServiceResponse,
    CodeFile,
    MessageType
)
from ...main_agent_four import programming_agent, handle_request  # Add this import
from containers.utils import DockerClientManager

logger = logging.getLogger(__name__)

class ServiceGenerator:
    """
    Generates service files and configurations for each service in a project
    by coordinating with Agent 4 for actual code generation.
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.model = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = StrOutputParser()
        self.protocol = AgentProtocol()
        self.docker_manager = DockerClientManager()

    async def generate_service(
        self,
        service_name: str,
        service_config: ServiceConfig,
        project_config: ProjectConfig
    ) -> Dict[str, str]:
        """
        Generate all files and configurations for a service based on its requirements.
        """
        logger.info(f"Starting service generation for: {service_name}")
        
        try:
            # Create service directory
            service_path = self.base_path / service_name
            service_path.mkdir(parents=True, exist_ok=True)

            # Step 1: Analyze requirements and create comprehensive specification
            service_spec = await self._analyze_service_requirements(
                service_name,
                service_config,
                project_config
            )

            # Step 2: Request code generation from Agent 4
            generated_code = await self._request_code_generation(service_spec)

            # Step 3: Generate all infrastructure and configuration
            generated_configs = await self._generate_configurations(
                service_name,
                service_config,
                project_config,
                service_spec
            )

            # Combine all generated content
            generated_files = {**generated_code, **generated_configs}

            # Step 4: Validate entire service
            validation_result = await self._validate_service(
                generated_files,
                service_spec
            )

            if not validation_result["is_valid"]:
                logger.error(f"Service validation failed: {validation_result['errors']}")
                raise ValueError(
                    f"Service validation failed: {validation_result['errors']}"
                )

            return generated_files

        except Exception as e:
            logger.error(f"Error generating service {service_name}: {str(e)}")
            raise

    async def _analyze_service_requirements(
        self,
        service_name: str,
        service_config: ServiceConfig,
        project_config: ProjectConfig
    ) -> Dict[str, Any]:
        """
        Analyze service requirements to create a comprehensive specification.
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze the service requirements and create a comprehensive specification that:
                1. Identifies all required components and their relationships
                2. Determines necessary configurations and infrastructure
                3. Specifies integration points and dependencies
                4. Defines security, monitoring, and operational requirements
                5. Outlines testing and validation criteria

                The specification should be:
                - Completely requirement-driven
                - Technology and tool agnostic
                - Implementation independent
                - Focused on capabilities and needs
                
                Do not make assumptions about specific:
                - Programming languages
                - Frameworks
                - Infrastructure tools
                - Deployment platforms"""),
                ("human", """Create a comprehensive service specification for:
                Service Name: {service_name}
                Service Configuration: {service_config}
                Project Configuration: {project_config}
                
                Return a detailed specification that can guide both code and infrastructure generation.""")
            ])

            chain = prompt | self.model | self.parser

            spec_str = await chain.ainvoke({
                "service_name": service_name,
                "service_config": service_config.dict(),
                "project_config": project_config.dict()
            })
            
            return json.loads(spec_str)

        except Exception as e:
            logger.error(f"Error analyzing service requirements: {str(e)}")
            raise

    async def _request_code_generation(self, service_spec: Dict[str, Any]) -> Dict[str, str]:
        """
        Request code generation from Agent 4 based on service specification.
        """
        try:
            # Create service request
            request = self.protocol.create_service_request(
                service_spec=service_spec,
                priority=CodeGenerationPriority.HIGH,
                requirements={
                    "optimization_targets": service_spec.get("optimization_targets", []),
                    "quality_requirements": service_spec.get("quality_requirements", {}),
                    "constraints": service_spec.get("constraints", {})
                }
            )
            
            if not self.protocol.validate_request(request):
                raise ValueError("Invalid service request")
            
            # Send to Agent 4
            response = await self._send_to_agent_4(request)
            
            if not self.protocol.validate_response(response):
                raise ValueError("Invalid response from Agent 4")
            
            return {
                file.path: file.content 
                for file in response.files
            }

        except Exception as e:
            logger.error(f"Error in code generation request: {str(e)}")
            raise

    async def _generate_configurations(
        self,
        service_name: str,
        service_config: ServiceConfig,
        project_config: ProjectConfig,
        service_spec: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate all configurations based on service requirements.
        This consolidated method handles all types of configurations:
        - Infrastructure (any required infrastructure tools)
        - Environment configurations
        - Security configurations
        - Monitoring and logging
        - Deployment configurations
        - CI/CD configurations
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Generate comprehensive configurations that fulfill all service requirements.
                Include configurations for any necessary:
                1. Infrastructure and deployment
                2. Service runtime environments
                3. Security measures and policies
                4. Monitoring, logging, and observability
                5. Scaling and performance
                6. Integration and communication
                7. Development and testing
                8. CI/CD and automation
                
                Important Guidelines:
                - Generate configurations based solely on requirements
                - Don't assume specific tools or platforms
                - Include all necessary components
                - Ensure configurations work together
                - Consider security and scalability
                - Enable observability and maintenance
                
                Return configurations in appropriate formats based on their purpose."""),
                ("human", """Generate all required configurations for:
                Service Name: {service_name}
                Service Specification: {service_spec}
                Service Configuration: {service_config}
                Project Configuration: {project_config}
                
                Return a dictionary mapping file paths to their contents.""")
            ])

            chain = prompt | self.model | self.parser

            configs = await chain.ainvoke({
                "service_name": service_name,
                "service_spec": service_spec,
                "service_config": service_config.dict(),
                "project_config": project_config.dict()
            })
            
            return json.loads(configs)

        except Exception as e:
            logger.error(f"Error generating configurations: {str(e)}")
            raise

    async def _validate_service(
        self,
        generated_files: Dict[str, str],
        service_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate the complete service including all generated files and configurations.
        """
        try:
            # Create validation request for Agent 4
            request = self.protocol.create_service_request(
                service_spec=service_spec,
                message_type=MessageType.CONFIG_REQUEST,
                files=generated_files,
                requirements={
                    "validation_criteria": service_spec.get("validation_criteria", {}),
                    "quality_requirements": service_spec.get("quality_requirements", {}),
                    "compliance_requirements": service_spec.get("compliance_requirements", {})
                }
            )
            
            # Get validation results from Agent 4
            response = await self._send_to_agent_4(request)
            
            return {
                "is_valid": response.status == "valid",
                "errors": response.warnings,
                "validation_details": response.additional_info,
                "suggested_improvements": response.additional_info.get("improvements", [])
            }

        except Exception as e:
            logger.error(f"Error validating service: {str(e)}")
            raise

    async def _send_to_agent_4(self, request: ServiceRequest) -> ServiceResponse:
        """
        Send request to Agent 4 and await response.
        Implements the communication protocol between ServiceGenerator and Agent 4.
        """
        try:
            # Send request to Agent 4
            response = await handle_request(request)
            
            if isinstance(response, dict) and response.get("status") == "queued":
                # Request was queued, wait for completion
                return await self._wait_for_agent4_completion(request.request_id)
            else:
                # Direct response
                return response
                
        except Exception as e:
            logger.error(f"Error communicating with Agent 4: {str(e)}")
            raise

    async def _wait_for_agent4_completion(self, request_id: str) -> ServiceResponse:
        """
        Wait for Agent 4 to complete processing a queued request.
        Implements polling with exponential backoff.
        """
        max_attempts = 10
        base_delay = 1  # seconds
        
        for attempt in range(max_attempts):
            try:
                # Check if task is complete
                if request_id in programming_agent.ongoing_tasks:
                    task = programming_agent.ongoing_tasks[request_id]
                    if task.done():
                        return await task
                
                # Calculate next delay with exponential backoff
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error waiting for Agent 4 completion: {str(e)}")
                raise
                
        raise TimeoutError(f"Request {request_id} timed out waiting for Agent 4")

