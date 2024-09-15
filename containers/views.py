from rest_framework import status
from rest_framework.generics import ListAPIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import UserContainer, Image
from .serializers import UserContainerSerializer,ImageSerializer
from profiles.models import UserProfile
import docker
import logging
from django.conf import settings
import datetime
from django.db import transaction
import time

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

            time.sleep(5)  # Increased wait time

            container_obj.reload()
            logger.info(f"Container status after start attempt: {container_obj.status}")

            if container_obj.status != 'running':
                logger.error(f'Container failed to start. Status: {container_obj.status}')
                logs = container_obj.logs().decode('utf-8')
                logger.error(f'Container logs: {logs}')
                return Response({'error': 'Container failed to start', 'logs': logs}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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

        except docker.errors.ContainerError as ce:
            logger.exception(f"Docker Container Error: {str(ce)}")
            return Response({'error': str(ce)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
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
                # Check if the container actually exists in Docker
                try:
                    docker_container = client.containers.get(container.container_id)
                    # Update container info if necessary
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
                    # Container doesn't exist in Docker, remove from DB and create a new one
                    logger.warning(f"Container {container.container_id} not found in Docker. Removing from DB and creating a new one.")
                    container.delete()
                    raise UserContainer.DoesNotExist()
            except UserContainer.DoesNotExist:
                # Create a new container
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

        except Image.DoesNotExist:
            logger.error(f"No image found with ID {image_id} for language {language}")
            return {'error': 'No image found corresponding to the provided language.'}
        except docker.errors.ContainerError as ce:
            logger.exception("Error managing Docker container operations")
            return {'error': str(ce)}
        except Exception as e:
            logger.exception("Unexpected error in get_or_create_container")
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
            # Check if the image actually exists in Docker
            try:
                docker_image = client.images.get(f"{image_in_db.repository}:{image_in_db.tag}")
                # Update image info if necessary
                if docker_image.id != image_in_db.image_id:
                    image_in_db.image_id = docker_image.id
                    image_in_db.save()
                logger.info(f"Using existing image: {image_in_db.image_id}")
            except docker.errors.ImageNotFound:
                # Image doesn't exist in Docker, remove from DB and pull a new one
                logger.warning(f"Image {image_in_db.image_id} not found in Docker. Removing from DB and pulling a new one.")
                image_in_db.delete()
                raise Image.DoesNotExist()
        except Image.DoesNotExist:
            try:
                logger.info(f"Pulling a new Image from Docker for {language_config['repository']}:{language_config['tag']}")
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
            except docker.errors.APIError as e:
                logger.exception(f"Failed to pull Docker image: {str(e)}")
                raise Exception(f"Failed to pull Docker image: {str(e)}")
            except Exception as e:
                logger.exception(f"Failed to add image record to DB: {str(e)}")
                raise Exception(f"Failed to create image record due to an error: {str(e)}")

        return image_in_db.image_id


