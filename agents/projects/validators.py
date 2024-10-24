import logging
from typing import Dict, List, Tuple
from .types import (
    ProjectConfig, 
    ServiceConfig, 
    NetworkConfig,
    ServiceType, 
    DeploymentType, 
    VolumeMount,
    HealthCheck,
    PortMapping,
)

logger = logging.getLogger(__name__)

class ProjectValidator:
    """Validates project configurations and ensures they meet requirements"""

    def __init__(self):
        self.required_services = {
            DeploymentType.PRODUCTION: {
                ServiceType.REVERSE_PROXY,
                ServiceType.MONITORING,
                ServiceType.LOGGING
            },
            DeploymentType.STAGING: {
                ServiceType.REVERSE_PROXY,
                ServiceType.MONITORING
            },
            DeploymentType.DEVELOPMENT: set()
        }

    async def validate_project_config(
        self,
        config: ProjectConfig,
        strict: bool = True
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete project configuration
        
        Args:
            config: ProjectConfig to validate
            strict: Whether to fail on warnings
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        try:
            # Validate basic project configuration
            if not config.services:
                errors.append("Project must have at least one service")

            # Check required services based on deployment type
            required = self.required_services[config.deployment_type]
            existing = {service.type for service in config.services.values()}
            missing = required - existing
            if missing:
                if strict:
                    errors.append(f"Missing required services for {config.deployment_type}: {missing}")
                else:
                    warnings.append(f"Missing recommended services for {config.deployment_type}: {missing}")

            # Validate services
            for service_name, service_config in config.services.items():
                service_errors, service_warnings = await self._validate_service(
                    service_name,
                    service_config,
                    config
                )
                errors.extend(service_errors)
                warnings.extend(service_warnings)

            # Validate networks
            network_errors, network_warnings = self._validate_networks(
                config.networks,
                config.services
            )
            errors.extend(network_errors)
            warnings.extend(network_warnings)

            # Validate resource allocation
            resource_errors, resource_warnings = self._validate_resources(
                config.services,
                config.deployment_type
            )
            errors.extend(resource_errors)
            warnings.extend(resource_warnings)

            # Validate dependencies
            dependency_errors = self._validate_dependencies(config.services)
            errors.extend(dependency_errors)

            return not errors if strict else True, errors, warnings

        except Exception as e:
            logger.error(f"Error validating project config: {str(e)}")
            return False, [f"Validation error: {str(e)}"], []

    async def _validate_service(
        self,
        service_name: str,
        service: ServiceConfig,
        project_config: ProjectConfig
    ) -> Tuple[List[str], List[str]]:
        """Validate a single service configuration"""
        errors = []
        warnings = []

        # Validate base image
        if not await self._validate_docker_image(service.base_image, service.tag):
            errors.append(f"Invalid or inaccessible Docker image: {service.base_image}:{service.tag}")

        # Validate port mappings
        port_errors = self._validate_ports(service.ports, project_config.services)
        errors.extend(port_errors)

        # Validate health checks
        if service.health_check:
            health_errors = self._validate_health_check(service.health_check)
            errors.extend(health_errors)
        elif service.type != ServiceType.FRONTEND:
            warnings.append(f"Service {service_name} has no health check configured")

        # Validate volume mounts
        volume_errors = self._validate_volumes(service.volumes)
        errors.extend(volume_errors)

        # Type-specific validations
        type_errors, type_warnings = self._validate_service_type_specific(
            service_name,
            service,
            project_config
        )
        errors.extend(type_errors)
        warnings.extend(type_warnings)

        return errors, warnings

    async def _validate_docker_image(self, image: str, tag: str) -> bool:
        """Validate that a Docker image exists and is accessible"""
        try:
            # You would typically check Docker Hub or your registry here
            # For now, we'll assume it exists
            return True
        except Exception as e:
            logger.error(f"Error validating Docker image: {str(e)}")
            return False

    def _validate_ports(
        self,
        ports: List[PortMapping],
        services: Dict[str, ServiceConfig]
    ) -> List[str]:
        """Validate port mappings"""
        errors = []
        used_ports = set()

        for port in ports:
            if port.host_port:
                if port.host_port in used_ports:
                    errors.append(f"Host port {port.host_port} is already in use")
                used_ports.add(port.host_port)

                if port.host_port < 1024 and port.is_public:
                    errors.append(f"Public port {port.host_port} is in privileged range")

        return errors

    def _validate_health_check(self, health_check: HealthCheck) -> List[str]:
        """Validate health check configuration"""
        errors = []

        if health_check.interval < health_check.timeout:
            errors.append("Health check interval must be greater than timeout")

        if health_check.timeout < 1:
            errors.append("Health check timeout must be positive")

        if health_check.retries < 1:
            errors.append("Health check retries must be positive")

        return errors

    def _validate_volumes(self, volumes: List[VolumeMount]) -> List[str]:
        """Validate volume mount configurations"""
        errors = []

        for volume in volumes:
            if volume.type not in ['bind', 'volume']:
                errors.append(f"Invalid volume type: {volume.type}")

            if volume.type == 'bind':
                if not volume.source.startswith('/'):
                    errors.append(f"Bind mount source must be absolute path: {volume.source}")

        return errors

    def _validate_service_type_specific(
        self,
        service_name: str,
        service: ServiceConfig,
        project_config: ProjectConfig
    ) -> Tuple[List[str], List[str]]:
        """Validate configurations specific to service types"""
        errors = []
        warnings = []

        if service.type == ServiceType.DATABASE:
            if not any(v.type == 'volume' for v in service.volumes):
                errors.append(f"Database service {service_name} must have persistent storage")

        elif service.type == ServiceType.FRONTEND:
            if not any(s.type == ServiceType.BACKEND for s in project_config.services.values()):
                warnings.append(f"Frontend service {service_name} has no backend service")

        elif service.type == ServiceType.BACKEND:
            if not service.health_check:
                errors.append(f"Backend service {service_name} must have a health check")

        return errors, warnings

    def _validate_networks(
        self,
        networks: List[NetworkConfig],
        services: Dict[str, ServiceConfig]
    ) -> Tuple[List[str], List[str]]:
        """Validate network configurations"""
        errors = []
        warnings = []

        if not networks:
            warnings.append("No custom networks defined, using default network")
            return errors, warnings

        network_names = {net.name for net in networks}
        if len(network_names) != len(networks):
            errors.append("Duplicate network names found")

        return errors, warnings

    def _validate_resources(
        self,
        services: Dict[str, ServiceConfig],
        deployment_type: DeploymentType
    ) -> Tuple[List[str], List[str]]:
        """Validate resource allocations for services"""
        errors = []
        warnings = []

        total_cpu = sum(s.resources.cpu_count for s in services.values())
        total_memory = sum(s.resources.memory_mb for s in services.values())

        # Resource limits based on deployment type
        resource_limits = {
            DeploymentType.PRODUCTION: {
                'max_cpu_per_service': 4.0,
                'min_memory_mb': 512,
                'max_memory_mb': 16384,
            },
            DeploymentType.STAGING: {
                'max_cpu_per_service': 2.0,
                'min_memory_mb': 256,
                'max_memory_mb': 8192,
            },
            DeploymentType.DEVELOPMENT: {
                'max_cpu_per_service': 1.0,
                'min_memory_mb': 128,
                'max_memory_mb': 4096,
            }
        }

        limits = resource_limits[deployment_type]

        for service_name, service in services.items():
            # CPU validation
            if service.resources.cpu_count > limits['max_cpu_per_service']:
                errors.append(
                    f"Service {service_name} CPU allocation ({service.resources.cpu_count}) "
                    f"exceeds maximum ({limits['max_cpu_per_service']})"
                )

            # Memory validation
            if service.resources.memory_mb < limits['min_memory_mb']:
                errors.append(
                    f"Service {service_name} memory allocation ({service.resources.memory_mb}MB) "
                    f"below minimum ({limits['min_memory_mb']}MB)"
                )
            elif service.resources.memory_mb > limits['max_memory_mb']:
                errors.append(
                    f"Service {service_name} memory allocation ({service.resources.memory_mb}MB) "
                    f"exceeds maximum ({limits['max_memory_mb']}MB)"
                )

        # Check total resource allocation
        if deployment_type != DeploymentType.DEVELOPMENT:
            if total_cpu > 8.0:
                warnings.append(f"Total CPU allocation ({total_cpu}) is quite high")
            if total_memory > limits['max_memory_mb'] * 2:
                warnings.append(f"Total memory allocation ({total_memory}MB) is quite high")

        return errors, warnings

    def _validate_dependencies(self, services: Dict[str, ServiceConfig]) -> List[str]:
        """Validate service dependencies"""
        errors = []
        dependency_graph = {}
        
        # Build dependency graph
        for service_name, service in services.items():
            dependencies = [dep.service_name for dep in service.dependencies]
            dependency_graph[service_name] = dependencies
            
            # Validate dependency existence
            for dep in dependencies:
                if dep not in services:
                    errors.append(f"Service {service_name} depends on non-existent service {dep}")

        # Check for circular dependencies
        try:
            self._detect_cycles(dependency_graph)
        except ValueError as e:
            errors.append(str(e))

        # Validate dependency conditions
        for service_name, service in services.items():
            for dep in service.dependencies:
                dependent_service = services.get(dep.service_name)
                if dependent_service:
                    if dep.condition == 'service_healthy' and not dependent_service.health_check:
                        errors.append(
                            f"Service {service_name} depends on health check of {dep.service_name}, "
                            "but no health check is configured"
                        )

        return errors

    def _detect_cycles(self, graph: Dict[str, List[str]]) -> None:
        """Detect cycles in dependency graph using DFS"""
        visited = set()
        path = set()

        def visit(node: str):
            if node in path:
                path_list = list(path)
                cycle_start = path_list.index(node)
                cycle = path_list[cycle_start:] + [node]
                raise ValueError(f"Circular dependency detected: {' -> '.join(cycle)}")
            
            if node in visited:
                return

            visited.add(node)
            path.add(node)

            for neighbor in graph.get(node, []):
                visit(neighbor)

            path.remove(node)

        for node in graph:
            if node not in visited:
                visit(node)

    async def get_recommended_fixes(
        self,
        config: ProjectConfig,
        errors: List[str],
        warnings: List[str]
    ) -> Dict[str, List[str]]:
        """Generate recommended fixes for validation issues"""
        fixes = {
            'errors': [],
            'warnings': []
        }

        # Common error patterns and their fixes
        error_fixes = {
            'must have persistent storage': 
                'Add a named volume to the service configuration',
            'must have a health check':
                'Configure a health check with appropriate interval and timeout values',
            'is already in use':
                'Use a different host port or remove the port mapping if not needed',
            'in privileged range':
                'Use a port number above 1024 or configure a reverse proxy',
            'Circular dependency':
                'Redesign service dependencies to remove circular references'
        }

        # Generate fixes for errors
        for error in errors:
            for pattern, fix in error_fixes.items():
                if pattern in error:
                    fixes['errors'].append(f"For error '{error}': {fix}")

        # Generate fixes for warnings
        for warning in warnings:
            if 'no health check configured' in warning:
                fixes['warnings'].append(
                    f"For warning '{warning}': Consider adding a health check "
                    "for better reliability monitoring"
                )
            elif 'No custom networks defined' in warning:
                fixes['warnings'].append(
                    f"For warning '{warning}': Consider defining custom networks "
                    "for better service isolation"
                )

        return fixes

class ProjectValidationError(Exception):
    """Custom exception for project validation errors"""
    def __init__(self, message: str, errors: List[str], warnings: List[str]):
        self.message = message
        self.errors = errors
        self.warnings = warnings
        super().__init__(self.message)