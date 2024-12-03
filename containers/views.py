from rest_framework import status
from rest_framework.generics import ListAPIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import UserContainer, Image, Project
from .serializers import UserContainerSerializer, ImageSerializer, ProjectSerializer
from profiles.models import UserProfile
from agents.technology_validator import TechnologyValidator
from langchain_core.prompts import ChatPromptTemplate
import docker
import logging
from django.conf import settings
import datetime
from django.db import transaction
import time
from .bot import generate_code, generate_application

from .bot import IntelligentApplicationGenerator
from typing import Dict, List
from django.http import HttpRequest
import os
from .utils import DockerClientManager
from agents.project_analyzer import ProjectArchitecture, ProjectAnalyzer

# Set up logging
logging.getLogger('docker').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

class ContainerListView(ListAPIView):
    queryset = UserContainer.objects.all()
    serializer_class = UserContainerSerializer
    permission_classes = [IsAuthenticated]

class ImageView(ListAPIView):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer
    permission_classes = [IsAuthenticated]

class ExecuteCodeView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        user_profile = UserProfile.objects.get(user=request.user)
        code = request.data.get('code', '')
        language = request.data.get('language', 'python')

        if not code:
            logger.error("No code provided for execution")
            return Response({'error': 'No code provided'}, status=status.HTTP_400_BAD_REQUEST)

        if language not in settings.LANGUAGE_DOCKER_IMAGES:
            logger.error(f"Unsupported language: {language}")
            return Response({'error': f'Unsupported language: {language}'}, status=status.HTTP_400_BAD_REQUEST)

        logger.info(f"Fetching Container to start Code execution for user {user_profile.user.username}")
        container_data = self.get_or_create_container(user_profile, language)
        if 'error' in container_data:
            logger.error(f"Container error: {container_data['error']}")
            return Response(container_data, status=status.HTTP_400_BAD_REQUEST)

        client = docker.from_env()
        try:
            container_obj = client.containers.get(container_data['container_id'])
            logger.info(f"Retrieved container {container_data['container_id']}")
            
            logger.info(f"Starting container {container_data['container_id']}")
            container_obj.start()

            time.sleep(5)  # Wait for container to start

            container_obj.reload()
            logger.info(f"Container status after start attempt: {container_obj.status}")

            if container_obj.status != 'running':
                logger.error(f'Container failed to start. Status: {container_obj.status}')
                logs = container_obj.logs().decode('utf-8')
                logger.error(f'Container logs: {logs}')
                return Response({
                    'error': 'Container failed to start', 
                    'logs': logs
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Construct the command based on the language configuration
            lang_config = settings.LANGUAGE_DOCKER_IMAGES[language]
            command = lang_config['default_command'].split()
            command.extend(lang_config.get('execution_flags', []))
            command.append(code)

            logger.info(f"Executing command: {command}")
            
            try:
                exec_output = container_obj.exec_run(command, stdout=True, stderr=True)
                if exec_output.exit_code == 0:
                    logger.info("Code execution successful")
                    return Response({'output': exec_output.output.decode('utf-8')}, status=status.HTTP_200_OK)
                else:
                    error_message = exec_output.output.decode('utf-8')
                    logger.error(f"Execution error: {error_message}")
                    return Response({'error': error_message}, status=status.HTTP_200_OK)
            finally:
                logger.info(f"Stopping container {container_data['container_id']}")
                container_obj.stop()

        except Exception as e:
            logger.exception(f"Unexpected Error: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get_or_create_container(self, user_profile, language):
        client = docker.from_env()
        try:
            image_id = self.get_or_create_image(language)
            image = Image.objects.get(pk=image_id)

            try:
                container = UserContainer.objects.get(user_profile=user_profile)
                try:
                    docker_container = client.containers.get(container.container_id)
                    if docker_container.name != container.container_name:
                        container.container_name = docker_container.name
                        container.save()
                    container_data = {
                        'container_id': container.container_id,
                        'container_name': container.container_name,
                        'default_command': image.default_command
                    }
                    logger.info(f"Using existing container: {container.container_id}")
                except docker.errors.NotFound:
                    logger.warning(f"Container {container.container_id} not found in Docker. Creating new one.")
                    container.delete()
                    raise UserContainer.DoesNotExist()
            except UserContainer.DoesNotExist:
                command = image.default_command if image.default_command else "No command set"
                container_name = f"user_container_{user_profile.user.id}"

                new_container = client.containers.create(
                    image=image.image_id,
                    name=container_name,
                    command=command,
                    detach=True,
                    tty=True
                )

                with transaction.atomic():
                    container = UserContainer.objects.create(
                        user_profile=user_profile,
                        container_id=new_container.id,
                        container_name=new_container.name,
                        image=image
                    )
                container_data = {
                    'container_id': container.container_id,
                    'container_name': container.container_name,
                    'default_command': command
                }
                logger.info(f"Created new container: {container.container_id}")

            return container_data

        except Exception as e:
            logger.exception("Error in get_or_create_container")
            return {'error': str(e)}

    @transaction.atomic
    def get_or_create_image(self, language_for_execution):
        client = docker.from_env()
        language_config = settings.LANGUAGE_DOCKER_IMAGES.get(language_for_execution)

        if not language_config:
            logger.error(f"Unsupported programming language: {language_for_execution}")
            raise ValueError(f"Unsupported programming language: {language_for_execution}")

        try:
            image_in_db = Image.objects.get(repository=language_config['repository'], tag=language_config['tag'])
            try:
                docker_image = client.images.get(f"{image_in_db.repository}:{image_in_db.tag}")
                if docker_image.id != image_in_db.image_id:
                    image_in_db.image_id = docker_image.id
                    image_in_db.save()
                logger.info(f"Using existing image: {image_in_db.image_id}")
            except docker.errors.ImageNotFound:
                logger.warning(f"Image {image_in_db.image_id} not found in Docker. Pulling new one.")
                image_in_db.delete()
                raise Image.DoesNotExist()
        except Image.DoesNotExist:
            try:
                logger.info(f"Pulling new Image for {language_config['repository']}:{language_config['tag']}")
                docker_image = client.images.pull(language_config['repository'], tag=language_config['tag'])
                image_in_db = Image.objects.create(
                    image_id=docker_image.id,
                    repository=language_config['repository'],
                    tag=language_config['tag'],
                    language_for_execution=language_for_execution,
                    execution_flags=','.join(language_config['execution_flags']),
                    default_command=language_config['default_command']
                )
                logger.info(f"Created new image record: {image_in_db.image_id}")
            except Exception as e:
                logger.exception(f"Failed to create image: {str(e)}")
                raise

        return image_in_db.image_id

class GenerateCodeView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        language = request.data.get('language')
        code_instructions = request.data.get('instructions')

        if not language or not code_instructions:
            return Response({'error': 'Language and instructions are required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            generated_code = generate_code(language, code_instructions)
            
            execute_request = HttpRequest()
            execute_request.method = 'POST'
            execute_request.user = request.user
            execute_request.data = {
                'code': generated_code,
                'language': language
            }
            
            execute_view = ExecuteCodeView()
            execute_response = execute_view.post(execute_request)

            if execute_response.status_code == 200:
                return Response({
                    'code': generated_code,
                    'output': execute_response.data.get('output')
                }, status=status.HTTP_200_OK)
            else:
                error_feedback = execute_response.data.get('error')
                corrected_code = generate_code(language, code_instructions, error_feedback)
                
                execute_request.data['code'] = corrected_code
                execute_response = execute_view.post(execute_request)
                
                if execute_response.status_code == 200:
                    return Response({
                        'code': corrected_code,
                        'output': execute_response.data.get('output')
                    }, status=status.HTTP_200_OK)
                else:
                    return Response({
                        'error': 'Failed to generate executable code',
                        'last_attempt': corrected_code,
                        'execution_error': execute_response.data.get('error')
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            logger.exception("Error in code generation and execution")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GenerateApplicationView(APIView):
    permission_classes = [IsAuthenticated]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.docker_manager = DockerClientManager()
        self.tech_validator = TechnologyValidator()
        self.project_analyzer = ProjectAnalyzer()

    async def post(self, request, *args, **kwargs):
        try:
            # Extract request data
            instructions = request.data.get('instructions')
            technologies = request.data.get('technologies', [])

            if not instructions:
                return Response(
                    {'error': 'Instructions are required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            if not technologies:
                return Response(
                    {'error': 'At least one technology is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get user profile
            user_profile = UserProfile.objects.get(user=request.user)

            # Validate technologies
            logger.info(f"Validating technology stack for user {user_profile.user.username}")
            validation_result = await self.tech_validator.validate_stack(technologies)
            
            if not validation_result['is_valid']:
                return Response({
                    'error': 'Invalid technology configuration',
                    'validation_errors': validation_result['errors']
                }, status=status.HTTP_400_BAD_REQUEST)

            # Analyze project architecture
            logger.info("Analyzing project architecture")
            project_architecture = await self.project_analyzer.analyze_project(
                validation_result['validated_technologies'],
                instructions
            )

            # Validate architecture
            architecture_warnings = self.project_analyzer.validate_architecture(project_architecture)
            if architecture_warnings:
                logger.warning(f"Architecture warnings: {architecture_warnings}")

            # Determine Docker configuration
            logger.info("Determining Docker configuration")
            docker_config = await self.project_analyzer.determine_docker_configuration(
                project_architecture
            )

            # Generate development instructions
            logger.info("Generating development instructions")
            dev_instructions = await self.project_analyzer.get_development_instructions(
                project_architecture,
                docker_config
            )

            # Create project directory and files
            project_name = f"project_{user_profile.user.id}_{int(time.time())}"
            result = await self._create_project(
                project_name,
                user_profile,
                project_architecture,
                docker_config
            )

            # Add development instructions to files
            result['files']['DEVELOPMENT.md'] = dev_instructions

            # Add architecture warnings if any
            if architecture_warnings:
                result['warnings'] = architecture_warnings

            logger.info(f"Application generation completed: {project_name}")
            return Response(result, status=status.HTTP_201_CREATED)

        except UserProfile.DoesNotExist:
            logger.error(f"User profile not found for user {request.user.username}")
            return Response(
                {'error': 'User profile not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )

        except Exception as e:
            logger.exception("Application generation failed")
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def _create_project(
        self,
        project_name: str,
        user_profile: UserProfile,
        architecture: ProjectArchitecture,
        docker_config: Dict
    ) -> Dict:
        """Create project files and start services"""
        try:
            # Create project structure
            project_dir = os.path.join(
                os.path.dirname(__file__), 
                'projects',
                project_name
            )
            os.makedirs(project_dir, exist_ok=True)

            # Set up services
            services_info = {}
            for service_name, service_config in architecture.service_configuration.items():
                service_info = await self._setup_service(
                    project_name,
                    service_name,
                    service_config,
                    docker_config[service_name],
                    project_dir,
                    user_profile
                )
                services_info[service_name] = service_info

            return {
                'message': 'Application generated and running',
                'project_name': project_name,
                'project_directory': project_dir,
                'services': services_info,
                'files': self._get_project_files(project_dir),
                'architecture': architecture.dict()
            }

        except Exception as e:
            logger.error(f"Error creating project: {str(e)}")
            self._cleanup_failed_project(project_name, project_dir)
            raise

    async def _setup_service(
        self,
        project_name: str,
        service_name: str,
        service_config: Dict,
        docker_config: Dict,
        project_dir: str,
        user_profile: UserProfile
    ) -> Dict:
        """Set up a single service"""
        try:
            service_dir = os.path.join(project_dir, service_name)
            os.makedirs(service_dir, exist_ok=True)

            # Create service files
            await self._create_service_files(service_dir, service_config)

            # Create database records
            with transaction.atomic():
                # Create image record
                image = Image.objects.create(
                    image_id=f"{project_name}_{service_name}",
                    repository=docker_config['base_image'].split(':')[0],
                    tag=docker_config['base_image'].split(':')[1],
                    language_for_execution=service_config.get('language')
                )

                # Start container
                container_info = await self._start_container(
                    service_dir,
                    docker_config,
                    f"{service_name}_{project_name}"
                )

                # Create container record
                UserContainer.objects.create(
                    container_id=container_info['container_id'],
                    user_profile=user_profile,
                    container_name=container_info['container_name'],
                    image=image
                )

                return {
                    'url': container_info['url'],
                    'status': container_info['status'],
                    'container_id': container_info['container_id']
                }

        except Exception as e:
            logger.error(f"Error setting up service {service_name}: {str(e)}")
            raise

    def _cleanup_failed_project(self, project_name: str, project_dir: str) -> None:
        """Clean up resources after a failed project creation"""
        try:
            # Clean up Docker resources
            self.docker_manager.cleanup_resources(project_name)

            # Clean up project directory
            if os.path.exists(project_dir):
                import shutil
                shutil.rmtree(project_dir)

            # Clean up database records
            with transaction.atomic():
                Project.objects.filter(project_name=project_name).delete()
                Image.objects.filter(image_id__startswith=project_name).delete()
                UserContainer.objects.filter(container_name__startswith=project_name).delete()

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    def _get_project_files(self, project_dir: str) -> Dict[str, str]:
        """Get all project files and their contents"""
        files_dict = {}
        for root, _, files in os.walk(project_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, project_dir)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files_dict[relative_path] = f.read()
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
                    files_dict[relative_path] = f"Error reading file: {str(e)}"
        
        return files_dict

    async def _create_service_files(self, service_dir: str, service_config: Dict) -> None:
        """Create all necessary files for a service"""
        try:
            # Generate service-specific files using LLM
            for file_config in service_config['files']:
                file_path = os.path.join(service_dir, file_config['path'])
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Generate file content based on type
                if file_config['type'] == 'code':
                    content = await self._generate_code_file(file_config, service_config)
                elif file_config['type'] == 'config':
                    content = await self._generate_config_file(file_config, service_config)
                elif file_config['type'] == 'docker':
                    content = await self._generate_docker_file(file_config, service_config)
                else:
                    content = file_config.get('content', '')

                # Write file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

        except Exception as e:
            logger.error(f"Error creating service files: {str(e)}")
            raise

    async def _generate_code_file(self, file_config: Dict, service_config: Dict) -> str:
        """Generate content for a code file using LLM"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert {service_config['language']} developer.
            Generate code for a {file_config['type']} file that:
            1. Implements the specified functionality
            2. Follows best practices and patterns
            3. Includes proper error handling
            4. Has comprehensive comments
            5. Is maintainable and testable
            
            Use these technologies: {service_config['technologies']}"""),
            ("human", f"""Create the content for:
            File: {file_config['path']}
            Purpose: {file_config['purpose']}
            Requirements: {file_config['requirements']}
            
            Generate production-ready code following best practices.""")
        ])

        chain = prompt | self.model | StrOutputParser()
        return await chain.ainvoke({})

    async def _generate_config_file(self, file_config: Dict, service_config: Dict) -> str:
        """Generate content for a configuration file"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a DevOps expert specializing in application configuration.
            Create a configuration file that:
            1. Follows the required format
            2. Includes all necessary settings
            3. Uses appropriate values
            4. Is properly documented"""),
            ("human", f"""Create a {file_config['format']} configuration file for:
            Purpose: {file_config['purpose']}
            Settings: {file_config['settings']}
            
            Generate a production-ready configuration file.""")
        ])

        chain = prompt | self.model | StrOutputParser()
        return await chain.ainvoke({})

    async def _generate_docker_file(self, file_config: Dict, service_config: Dict) -> str:
        """Generate content for a Dockerfile"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Docker expert.
            Create a Dockerfile that:
            1. Uses appropriate base image
            2. Follows best practices
            3. Optimizes for size and security
            4. Includes proper health checks
            5. Sets up the environment correctly"""),
            ("human", f"""Create a Dockerfile for:
            Language: {service_config['language']}
            Dependencies: {service_config.get('dependencies', [])}
            Requirements: {file_config.get('requirements', {})}
            
            Generate a production-ready Dockerfile.""")
        ])

        chain = prompt | self.model | StrOutputParser()
        return await chain.ainvoke({})

    async def _start_container(self, service_dir: str, docker_config: Dict, container_name: str) -> Dict:
        """Start a Docker container for a service"""
        try:
            # Build image
            logger.info(f"Building image for {container_name}")
            image_id = await self.docker_manager.build_image(
                service_dir,
                f"{container_name}:latest",
                docker_config
            )

            # Start container
            logger.info(f"Starting container {container_name}")
            container = await self.docker_manager.start_container(
                image_id,
                container_name,
                docker_config.get('environment', {}),
                docker_config.get('ports', {}),
                docker_config.get('volumes', []),
                docker_config.get('network_mode', 'bridge')
            )

            # Wait for container to be ready
            logger.info("Waiting for container to be ready")
            is_ready = await self.docker_manager.wait_for_container(
                container,
                docker_config.get('healthcheck', {})
            )

            if not is_ready:
                raise Exception(f"Container {container_name} failed to start properly")

            # Get container info
            container_info = await self.docker_manager.get_container_info(container)
            
            return {
                'container_id': container.id,
                'container_name': container_name,
                'url': self._get_service_url(container_info['ports'], docker_config),
                'status': container_info['status']
            }

        except Exception as e:
            logger.error(f"Error starting container: {str(e)}")
            raise

    def _get_service_url(self, port_mappings: Dict, docker_config: Dict) -> str:
        """Generate service URL from port mappings"""
        main_port = docker_config.get('main_port')
        if main_port and f"{main_port}/tcp" in port_mappings:
            host_port = port_mappings[f"{main_port}/tcp"][0]['HostPort']
            protocol = docker_config.get('protocol', 'http')
            return f"{protocol}://localhost:{host_port}"
        return None