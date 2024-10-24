import os
import json
import time
import docker
import logging
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from .utils import DockerClientManager
from django.conf import settings


load_dotenv()

logger = logging.getLogger(__name__)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY environment variable")

def extract_code(text, language):
    """
    Extract code from text using an LLM.
    
    :param text: The text containing the code
    :param language: The programming language of the code
    :return: Extracted code
    """
    model = ChatOpenAI(model="gpt-3.5-turbo")
    parser = StrOutputParser()

    system_template = (
        "You are an expert code extractor. Given the following text, "
        "extract only the {language} code. Do not include any explanations, "
        "comments, or markdown formatting. Return only the executable code."
    )
    
    human_template = "Extract the {language} code from this text:\n\n{text}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])

    chain = prompt_template | model | parser
    result = chain.invoke({"language": language, "text": text})

    return result.strip()

def generate_code(language, code_instructions, error_feedback=None):
    """
    Generate a code snippet based on the given instructions and language.
    
    :param language: The programming language to use
    :param code_instructions: The instructions for generating the code
    :param error_feedback: Any error feedback from previous attempts (optional)
    :return: Generated code snippet
    """
    model = ChatOpenAI(model="gpt-4")
    parser = StrOutputParser()

    if error_feedback:
        system_template = (
            "Generate only executable {language} code for the following instructions. "
            "Do not include any explanations or markdown formatting. "
            "Previous attempt resulted in the following error: {error_feedback}. "
            "Please correct the code to avoid this error."
        )
    else:
        system_template = (
            "Generate only executable {language} code for the following instructions. "
            "Do not include any explanations or markdown formatting."
        )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{code}")
    ])
    
    chain = prompt_template | model | parser
    result = chain.invoke({"language": language, "code": code_instructions, "error_feedback": error_feedback})
    
    return extract_code(result, language)

class ProjectPlanner:
    def __init__(self, model: ChatOpenAI):
        self.model = model

    def create_project_plan(self, technologies: List[Dict], instructions: str) -> Dict:
        """Generate a comprehensive project plan based on technologies and requirements"""
        system_prompt = """You are a senior software architect specializing in microservices and modern application design.
        Create a detailed project plan that includes:
        1. Overall architecture design
        2. Directory structure for each component
        3. Required files and their purposes
        4. Dependencies and configurations
        5. Docker setup requirements
        6. Integration points between services
        7. Development and production considerations
        8. Required environment variables
        9. Port mappings and networking
        
        Provide the plan in a structured JSON format that can be parsed and used programmatically.
        """
        
        tech_summary = "\n".join([
            f"- {tech['role']}: {tech['name']}" + (f" version {tech['version']}" if tech['version'] else "")
            for tech in technologies
        ])
        
        human_prompt = f"""Create a detailed project plan for an application with the following technologies:

                {tech_summary}

                Project Requirements:
                {instructions}

                Ensure the plan addresses all integration points between different technologies and includes
                complete Docker configuration requirements."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.model(messages)
        return json.loads(response.content)

    def generate_dockerfile(self, service_config: Dict) -> str:
        """Generate a Dockerfile for a specific service"""
        system_prompt = """You are an expert in Docker container configuration.
        Create a Dockerfile that follows best practices for the given technology stack.
        Include:
        - Appropriate base image
        - Multi-stage builds when beneficial
        - Security considerations
        - Environment setup
        - Proper CMD/ENTRYPOINT
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Generate a Dockerfile for a service with the following configuration:
            {json.dumps(service_config, indent=2)}
            """)
        ]
        
        response = self.model(messages)
        return response.content

class IntelligentApplicationGenerator:
    def __init__(self):
        self.model = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.planner = ProjectPlanner(self.model)
        self.docker_client = docker.from_env()
        self.debug_history = []

    def generate_application(
        self, 
        technologies: List[Dict], 
        instructions: str,
        user_id: str
    ) -> Dict:
        """Generate a multi-service application based on specified technologies"""
        try:
            # Step 1: Generate project plan
            logger.info("Generating project plan...")
            project_plan = self.planner.create_project_plan(technologies, instructions)
            
            # Step 2: Create project structure
            project_name = f"project_{user_id}_{int(time.time())}"
            project_dir = self._create_project_directory(project_name)
            
            # Step 3: Generate service files
            logger.info("Generating service files...")
            files_dict = self._generate_service_files(project_plan, project_dir)
            
            # Step 4: Set up Docker infrastructure
            logger.info("Setting up Docker infrastructure...")
            services_info = self._setup_docker_services(project_plan, project_dir)
            
            return {
                "project_name": project_name,
                "project_directory": project_dir,
                "files": files_dict,
                "services": services_info
            }
            
        except Exception as e:
            logger.error(f"Application generation failed: {str(e)}")
            self._cleanup_resources(project_dir)
            raise
    
    def _create_project_directory(self, project_name: str) -> str:
        """Create the project directory structure"""
        base_dir = os.path.join(os.path.dirname(__file__), 'projects')
        project_dir = os.path.join(base_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)
        return project_dir

    def _generate_service_files(self, project_plan: Dict, project_dir: str) -> Dict[str, str]:
        """Generate all service files based on the project plan"""
        files_dict = {}
        
        for service in project_plan['services']:
            service_dir = os.path.join(project_dir, service['name'])
            os.makedirs(service_dir, exist_ok=True)
            
            # Generate files for this service
            for file_config in service['files']:
                file_path = os.path.join(service_dir, file_config['path'])
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Generate file content using LLM
                content = self._generate_file_content(file_config, service)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Store in files_dict with relative path
                relative_path = os.path.relpath(file_path, project_dir)
                files_dict[relative_path] = content
                
            # Generate Dockerfile for the service
            dockerfile_content = self.planner.generate_dockerfile(service)
            dockerfile_path = os.path.join(service_dir, 'Dockerfile')
            
            with open(dockerfile_path, 'w', encoding='utf-8') as f:
                f.write(dockerfile_content)
            
            files_dict[f"{service['name']}/Dockerfile"] = dockerfile_content
        
        # Generate docker-compose.yml in project root
        docker_compose = self._generate_docker_compose(project_plan)
        docker_compose_path = os.path.join(project_dir, 'docker-compose.yml')
        
        with open(docker_compose_path, 'w', encoding='utf-8') as f:
            f.write(docker_compose)
        
        files_dict['docker-compose.yml'] = docker_compose
        
        return files_dict

    def _generate_file_content(self, file_config: Dict, service: Dict) -> str:
        """Generate content for a specific file using LLM"""
        system_prompt = f"""You are an expert developer in {service['technology']}.
        Generate the content for a {file_config['type']} file with the following requirements:
        - File path: {file_config['path']}
        - Purpose: {file_config['purpose']}
        - Dependencies: {', '.join(file_config.get('dependencies', []))}
        
        Follow best practices and include comprehensive error handling.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate the content for this file according to the project requirements.")
        ]
        
        response = self.model(messages)
        return response.content

    def _generate_docker_compose(self, project_plan: Dict) -> str:
        """Generate docker-compose.yml content"""
        system_prompt = """You are an expert in Docker and container orchestration.
        Create a docker-compose.yml file that:
        - Properly configures all services
        - Sets up networking between services
        - Manages environment variables
        - Configures volumes and persistence
        - Sets up health checks
        - Includes proper restart policies
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Generate a docker-compose.yml file for a project with the following configuration:
            {json.dumps(project_plan, indent=2)}
            """)
        ]
        
        response = self.model(messages)
        return response.content

    def _setup_docker_services(self, project_plan: Dict, project_dir: str) -> Dict[str, Dict]:
        """Build and start Docker services"""
        services_info = {}
        
        try:
            # Build images for each service
            for service in project_plan['services']:
                service_dir = os.path.join(project_dir, service['name'])
                image_name = f"{service['name']}:latest"
                
                # Build the image
                logger.info(f"Building image for {service['name']}...")
                self.docker_client.images.build(
                    path=service_dir,
                    tag=image_name,
                    rm=True,
                    forcerm=True
                )
                
                # Start the container
                logger.info(f"Starting container for {service['name']}...")
                container = self.docker_client.containers.run(
                    image_name,
                    name=f"{service['name']}_{int(time.time())}",
                    detach=True,
                    environment=service.get('environment', {}),
                    ports=service.get('ports', {}),
                    network_mode='bridge'
                )
                
                # Wait for container to be ready
                time.sleep(5)
                container.reload()
                
                # Get the mapped ports
                port_mappings = container.ports
                
                # Store service info
                services_info[service['name']] = {
                    'url': self._get_service_url(service, port_mappings),
                    'status': container.status,
                    'container_id': container.id
                }
        
        except Exception as e:
            logger.error(f"Error setting up Docker services: {str(e)}")
            self._cleanup_resources(project_dir)
            raise
        
        return services_info

    def _get_service_url(self, service: Dict, port_mappings: Dict) -> str:
        """Generate the service URL based on mapped ports"""
        main_port = service.get('main_port')
        if main_port and port_mappings and f"{main_port}/tcp" in port_mappings:
            host_port = port_mappings[f"{main_port}/tcp"][0]['HostPort']
            return f"http://localhost:{host_port}"
        return None

    def _cleanup_resources(self, project_dir: str) -> None:
        """Clean up resources in case of failure"""
        try:
            # Stop and remove containers
            containers = self.docker_client.containers.list(
                filters={'label': f'project_dir={project_dir}'}
            )
            for container in containers:
                try:
                    container.stop(timeout=10)
                    container.remove(force=True)
                except Exception as e:
                    logger.error(f"Error cleaning up container {container.id}: {str(e)}")
            
            # Remove images
            images = self.docker_client.images.list(
                filters={'label': f'project_dir={project_dir}'}
            )
            for image in images:
                try:
                    self.docker_client.images.remove(image.id, force=True)
                except Exception as e:
                    logger.error(f"Error cleaning up image {image.id}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def generate_application(language: str, instructions: str, base_image: str) -> Dict[str, str]:
    """
    Main entry point for application generation.
    
    Args:
        language (str): Programming language for the application
        instructions (str): User instructions for application generation
        base_image (str): Docker base image to use
        
    Returns:
        Dict[str, str]: Dictionary mapping file paths to their contents
        
    Raises:
        ValueError: If language or base_image is not supported
        Exception: If application generation fails
    """
    logger.info(f"Starting application generation for {language}")
    
    if not language or not instructions or not base_image:
        raise ValueError("Language, instructions, and base_image are required")
        
    if language not in settings.LANGUAGE_DOCKER_IMAGES:
        raise ValueError(f"Unsupported language: {language}")
        
    try:
        generator = IntelligentApplicationGenerator()
        files_dict = generator.generate_application(language, instructions, base_image)
        
        if not files_dict:
            raise Exception("No files were generated")
            
        logger.info(f"Successfully generated application with {len(files_dict)} files")
        return files_dict
        
    except Exception as e:
        logger.error(f"Application generation failed: {str(e)}")
        logger.debug("Failed attempt details:", exc_info=True)
        raise Exception(f"Failed to generate application: {str(e)}") from e