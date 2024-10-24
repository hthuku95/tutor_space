from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Optional
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger(__name__)

class ProjectArchitecture(BaseModel):
    """Represents the analyzed project architecture"""
    project_type: str = Field(description="Primary project type/framework")
    architecture_pattern: str = Field(description="Architectural pattern to use")
    service_configuration: Dict = Field(description="Service configuration details")
    integration_points: List[Dict] = Field(description="Integration points between services")
    infrastructure_requirements: Dict = Field(description="Required infrastructure components")

class ProjectAnalyzer:
    """Analyzes technology stack and determines project configuration using LLMs"""
    
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=ProjectArchitecture)
        
        # Prompt for analyzing project architecture
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert software architect specializing in modern application design.
            Analyze the provided technology stack and requirements to determine:
            1. Primary project type and framework choices
            2. Most appropriate architectural pattern
            3. Service configuration for each component
            4. Integration points between services
            5. Infrastructure requirements
            
            Consider:
            - Scalability requirements
            - Development efficiency
            - Modern best practices
            - Team productivity
            - Maintenance considerations
            
            Provide detailed, practical configuration that can be used to set up the development environment."""),
            ("human", """Analyze the following project requirements:
            
            Technologies:
            {technologies}
            
            Requirements:
            {requirements}
            
            Provide a detailed project architecture analysis in the specified JSON format.""")
        ])
        
        self.analysis_chain = self.analysis_prompt | self.model | self.parser

    async def analyze_project(self, technologies: List[Dict], requirements: str) -> ProjectArchitecture:
        """
        Analyze the project requirements and technology stack to determine project configuration
        
        Args:
            technologies: List of technology dictionaries with name, role, and version
            requirements: Project requirements and specifications
            
        Returns:
            ProjectArchitecture object containing detailed project configuration
        """
        try:
            # Format technologies for prompt
            tech_summary = "\n".join([
                f"- {tech['role'].upper()}: {tech['name']} " +
                f"(version: {tech['version']})" if tech.get('version') else f"- {tech['role'].upper()}: {tech['name']}"
                for tech in technologies
            ])
            
            # Get project analysis
            analysis = await self.analysis_chain.ainvoke({
                "technologies": tech_summary,
                "requirements": requirements
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing project architecture: {str(e)}")
            raise

    async def determine_docker_configuration(
        self,
        project_architecture: ProjectArchitecture
    ) -> Dict:
        """
        Determine Docker configuration based on project architecture
        
        Args:
            project_architecture: Analyzed project architecture
            
        Returns:
            Dictionary containing Docker configuration for each service
        """
        docker_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a DevOps expert specializing in Docker and container orchestration.
            Create a detailed Docker configuration that:
            1. Uses appropriate base images for each service
            2. Configures proper networking between services
            3. Sets up volumes and persistence
            4. Manages environment variables
            5. Implements health checks
            6. Establishes proper startup order
            7. Optimizes for both development and production
            
            Focus on creating a robust, scalable configuration that follows best practices."""),
            ("human", """Create Docker configuration for the following architecture:
            {architecture}
            
            Provide configuration in JSON format including:
            - Base images
            - Build contexts
            - Environment variables
            - Port mappings
            - Volume configurations
            - Network settings
            - Health checks
            - Dependencies""")
        ])
        
        docker_chain = docker_prompt | self.model | JsonOutputParser()
        
        try:
            docker_config = await docker_chain.ainvoke({
                "architecture": project_architecture.dict()
            })
            
            return docker_config
        
        except Exception as e:
            logger.error(f"Error determining Docker configuration: {str(e)}")
            raise

    def validate_architecture(self, architecture: ProjectArchitecture) -> List[str]:
        """
        Validate the generated architecture for common issues
        
        Args:
            architecture: ProjectArchitecture to validate
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        # Check for required components
        if not architecture.service_configuration:
            warnings.append("No service configuration specified")
            
        if not architecture.integration_points:
            warnings.append("No integration points defined between services")
            
        # Validate infrastructure requirements
        if not architecture.infrastructure_requirements:
            warnings.append("No infrastructure requirements specified")
            
        # Check for database configurations
        has_database = any(
            service.get('type') == 'database' 
            for service in architecture.service_configuration.values()
        )
        if not has_database:
            warnings.append("No database service configured")
            
        # Validate service dependencies
        for service_name, service_config in architecture.service_configuration.items():
            if 'dependencies' in service_config:
                for dependency in service_config['dependencies']:
                    if dependency not in architecture.service_configuration:
                        warnings.append(f"Service {service_name} depends on undefined service {dependency}")
        
        return warnings

    async def get_development_instructions(
        self,
        project_architecture: ProjectArchitecture,
        docker_config: Dict
    ) -> str:
        """
        Generate development setup instructions
        
        Args:
            project_architecture: Analyzed project architecture
            docker_config: Docker configuration
            
        Returns:
            Markdown-formatted setup instructions
        """
        instructions_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technical documentation expert.
            Create clear, step-by-step instructions for setting up the development environment.
            Include:
            1. Prerequisites and dependencies
            2. Environment setup steps
            3. Build and run instructions
            4. Development workflow
            5. Common issues and solutions
            
            Format the instructions in Markdown with clear sections and code examples."""),
            ("human", """Create setup instructions for:
            
            Architecture:
            {architecture}
            
            Docker Configuration:
            {docker_config}
            
            Provide detailed, practical instructions that a developer can follow.""")
        ])
        
        instruction_chain = instructions_prompt | self.model | StrOutputParser()
        
        try:
            instructions = await instruction_chain.ainvoke({
                "architecture": project_architecture.dict(),
                "docker_config": docker_config
            })
            
            return instructions
            
        except Exception as e:
            logger.error(f"Error generating development instructions: {str(e)}")
            raise