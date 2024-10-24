import os
import asyncio
import logging
import docker
from typing import Dict, List, Optional, AsyncIterator
from docker.errors import DockerException
from contextlib import asynccontextmanager
import aiodocker
import aiohttp
import backoff
from pathlib import Path

logger = logging.getLogger(__name__)

class DockerClientManager:
    """Enhanced Docker client manager with async support and improved container management"""

    def __init__(self):
        self.sync_client = None
        self.async_client = None
        self._initialize_sync_client()

    def _initialize_sync_client(self):
        """Initialize synchronous Docker client"""
        try:
            self.sync_client = docker.from_env()
            self.sync_client.ping()
            logger.info("Successfully connected to Docker daemon (sync)")
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {str(e)}")
            raise DockerException("Could not establish Docker connection")

    async def _get_async_client(self):
        """Get or create async Docker client"""
        if not self.async_client:
            try:
                self.async_client = aiodocker.Docker()
                logger.info("Successfully connected to Docker daemon (async)")
            except Exception as e:
                logger.error(f"Failed to connect to async Docker client: {str(e)}")
                raise
        return self.async_client

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def build_image(self, context_path: str, tag: str, config: Dict) -> str:
        """Build a Docker image with retries"""
        client = await self._get_async_client()
        
        try:
            logger.info(f"Building image {tag} from {context_path}")
            
            # Prepare build context
            tar_context = await self._create_tar_context(context_path)
            
            # Build image
            build_source = await client.images.build(
                fileobj=tar_context,
                encoding='utf-8',
                tag=tag,
                rm=True,
                forcerm=True,
                pull=True,
                buildargs=config.get('build_args', {}),
                labels={
                    'project_dir': context_path,
                    'managed_by': 'tutor_space'
                }
            )
            
            image_id = build_source.id
            logger.info(f"Successfully built image {tag} ({image_id})")
            return image_id
            
        except Exception as e:
            logger.error(f"Error building image {tag}: {str(e)}")
            raise

    async def _create_tar_context(self, context_path: str) -> AsyncIterator[bytes]:
        """Create a tar archive of the build context"""
        import tarfile
        import io
        
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            # Add .dockerignore if it exists
            dockerignore = Path(context_path) / '.dockerignore'
            if dockerignore.exists():
                tar.add(dockerignore, arcname='.dockerignore')
            
            for root, _, files in os.walk(context_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, context_path)
                    tar.add(file_path, arcname=arcname)
        
        tar_buffer.seek(0)
        while True:
            chunk = tar_buffer.read(8192)
            if not chunk:
                break
            yield chunk

    @asynccontextmanager
    async def start_container(
        self,
        image_id: str,
        container_name: str,
        environment: Dict,
        ports: Dict,
        volumes: List[str],
        network_mode: str = 'bridge',
        health_check: Optional[Dict] = None,
        resource_limits: Optional[Dict] = None
    ):
        """Start and manage a Docker container"""
        client = await self._get_async_client()
        container = None
        
        try:
            # Prepare host config
            host_config = {
                'PortBindings': self._format_port_bindings(ports),
                'NetworkMode': network_mode,
                'Binds': volumes
            }
            
            # Add resource limits if specified
            if resource_limits:
                host_config.update({
                    'Memory': resource_limits.get('memory'),
                    'NanoCPUs': resource_limits.get('cpu_count'),
                    'MemorySwap': resource_limits.get('memory_swap', -1)
                })
            
            # Create container
            config = {
                'Image': image_id,
                'name': container_name,
                'Environment': [f"{k}={v}" for k, v in environment.items()],
                'HostConfig': host_config,
                'Labels': {
                    'container_name': container_name,
                    'managed_by': 'tutor_space'
                }
            }
            
            # Add health check if specified
            if health_check:
                config['Healthcheck'] = {
                    'Test': health_check['test'],
                    'Interval': health_check.get('interval', 30000000000),
                    'Timeout': health_check.get('timeout', 30000000000),
                    'Retries': health_check.get('retries', 3)
                }
            
            container = await client.containers.create(**config)
            
            # Start container
            await container.start()
            logger.info(f"Started container {container_name}")
            
            yield container
            
        except Exception as e:
            logger.error(f"Error managing container {container_name}: {str(e)}")
            if container:
                await self._cleanup_container(container)
            raise
            
        finally:
            if container:
                await self._cleanup_container(container)

    def _format_port_bindings(self, ports: Dict) -> Dict:
        """Format port bindings for Docker API"""
        bindings = {}
        for container_port, host_port in ports.items():
            if isinstance(container_port, int):
                container_port = f"{container_port}/tcp"
            bindings[container_port] = [{'HostPort': str(host_port)}]
        return bindings

    async def wait_for_container(
        self,
        container: aiodocker.containers.Container,
        healthcheck: Dict,
        timeout: int = 30
    ) -> bool:
        """Wait for container to be healthy"""
        if not healthcheck:
            await asyncio.sleep(2)  # Basic delay for containers without healthcheck
            return True
            
        endpoint = healthcheck.get('endpoint', '/')
        expected_status = healthcheck.get('status', 200)
        
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                # Get container info
                container_info = await container.show()
                if container_info['State']['Health']['Status'] == 'healthy':
                    return True
                    
                # Try health endpoint
                if healthcheck.get('http'):
                    ports = container_info['NetworkSettings']['Ports']
                    host_port = None
                    for container_port, bindings in ports.items():
                        if bindings:
                            host_port = bindings[0]['HostPort']
                            break
                            
                    if host_port:
                        url = f"http://localhost:{host_port}{endpoint}"
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url) as response:
                                if response.status == expected_status:
                                    return True
                                    
            except Exception as e:
                logger.debug(f"Health check failed: {str(e)}")
                
            await asyncio.sleep(1)
            
        return False

    async def get_container_info(
        self,
        container: aiodocker.containers.Container
    ) -> Dict:
        """Get container information"""
        try:
            info = await container.show()
            return {
                'id': info['Id'],
                'name': info['Name'].lstrip('/'),
                'status': info['State']['Status'],
                'ports': info['NetworkSettings']['Ports'],
                'health': info['State'].get('Health', {}).get('Status'),
                'created': info['Created'],
                'platform': info['Platform'],
                'resource_usage': await self._get_container_stats(container)
            }
        except Exception as e:
            logger.error(f"Error getting container info: {str(e)}")
            raise

    async def _get_container_stats(
        self,
        container: aiodocker.containers.Container
    ) -> Dict:
        """Get container resource usage statistics"""
        try:
            stats = await container.stats(stream=False)
            return {
                'cpu_usage': stats['cpu_stats']['cpu_usage']['total_usage'],
                'memory_usage': stats['memory_stats']['usage'],
                'network_rx': stats['networks']['eth0']['rx_bytes'],
                'network_tx': stats['networks']['eth0']['tx_bytes']
            }
        except Exception as e:
            logger.error(f"Error getting container stats: {str(e)}")
            return {}

    async def _cleanup_container(self, container: aiodocker.containers.Container) -> None:
        """Clean up a container"""
        try:
            await container.stop()
            await container.delete()
        except Exception as e:
            logger.error(f"Error cleaning up container: {str(e)}")

    async def cleanup_resources(self, project_name: str) -> None:
        """Clean up all resources associated with a project"""
        client = await self._get_async_client()
        
        try:
            # Stop and remove containers
            containers = await client.containers.list(
                filters={'label': f'container_name={project_name}*'}
            )
            for container in containers:
                await self._cleanup_container(container)
                
            # Remove images
            images = await client.images.list(
                filters={'label': f'project_dir={project_name}*'}
            )
            for image in images:
                await client.images.delete(image['Id'], force=True)
                
            # Clean up networks
            networks = await client.networks.list(
                filters={'label': f'project={project_name}'}
            )
            for network in networks:
                await network.delete()
                
            # Clean up volumes
            volumes = await client.volumes.list(
                filters={'label': f'project={project_name}'}
            )
            for volume in volumes:
                await volume.delete()
                
        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")
            raise

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.async_client:
            await self.async_client.close()
        if self.sync_client:
            try:
                self.sync_client.close()
            except:
                pass