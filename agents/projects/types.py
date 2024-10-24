from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime

class ServiceType(str, Enum):
    """Types of services that can be part of a project"""
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    REVERSE_PROXY = "reverse_proxy"
    LOAD_BALANCER = "load_balancer"
    MONITORING = "monitoring"
    LOGGING = "logging"
    AUTHENTICATION = "authentication"
    STORAGE = "storage"

class DeploymentType(str, Enum):
    """Types of deployment configurations"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ResourceRequirements(BaseModel):
    """Resource requirements for a service"""
    cpu_count: Optional[float] = Field(default=1.0, description="Number of CPU cores")
    memory_mb: Optional[int] = Field(default=512, description="Memory in megabytes")
    disk_gb: Optional[int] = Field(default=10, description="Disk space in gigabytes")
    memory_swap_mb: Optional[int] = Field(default=-1, description="Swap memory in megabytes, -1 for unlimited")

class PortMapping(BaseModel):
    """Port mapping configuration"""
    container_port: int = Field(..., description="Port inside the container")
    host_port: Optional[int] = Field(None, description="Port on the host machine")
    protocol: str = Field(default="tcp", description="Protocol (tcp/udp)")
    is_public: bool = Field(default=False, description="Whether the port should be publicly accessible")

class HealthCheck(BaseModel):
    """Health check configuration"""
    test: List[str] = Field(..., description="Health check command")
    interval: int = Field(default=30, description="Interval between checks in seconds")
    timeout: int = Field(default=30, description="Timeout for each check in seconds")
    retries: int = Field(default=3, description="Number of retries before considered unhealthy")
    start_period: int = Field(default=0, description="Start period in seconds")

class VolumeMount(BaseModel):
    """Volume mount configuration"""
    source: str = Field(..., description="Source path or volume name")
    target: str = Field(..., description="Target path in container")
    type: str = Field(default="bind", description="Mount type (bind/volume)")
    read_only: bool = Field(default=False, description="Whether the mount is read-only")

class DependencyConfig(BaseModel):
    """Configuration for service dependencies"""
    service_name: str = Field(..., description="Name of the dependent service")
    required: bool = Field(default=True, description="Whether the dependency is required")
    condition: str = Field(default="service_healthy", description="Condition for dependency")
    timeout: int = Field(default=30, description="Timeout for dependency in seconds")

class ServiceConfig(BaseModel):
    """Configuration for a single service"""
    name: str = Field(..., description="Service name")
    type: ServiceType = Field(..., description="Type of service")
    base_image: str = Field(..., description="Base Docker image")
    tag: str = Field(default="latest", description="Image tag")
    build_context: str = Field(..., description="Build context directory")
    dockerfile_path: Optional[str] = Field(None, description="Custom Dockerfile path")
    command: Optional[List[str]] = Field(None, description="Command to run")
    entrypoint: Optional[List[str]] = Field(None, description="Container entrypoint")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    ports: List[PortMapping] = Field(default_factory=list, description="Port mappings")
    volumes: List[VolumeMount] = Field(default_factory=list, description="Volume mounts")
    health_check: Optional[HealthCheck] = Field(None, description="Health check configuration")
    resources: ResourceRequirements = Field(
        default_factory=ResourceRequirements,
        description="Resource requirements"
    )
    dependencies: List[DependencyConfig] = Field(
        default_factory=list,
        description="Service dependencies"
    )

    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment variables"""
        # Convert all values to strings
        return {k: str(v) for k, v in v.items()}

class NetworkConfig(BaseModel):
    """Network configuration"""
    name: str = Field(..., description="Network name")
    driver: str = Field(default="bridge", description="Network driver")
    internal: bool = Field(default=False, description="Whether the network is internal")
    enable_ipv6: bool = Field(default=False, description="Whether to enable IPv6")
    attachable: bool = Field(default=True, description="Whether the network is attachable")

class ProjectConfig(BaseModel):
    """Complete project configuration"""
    project_name: str = Field(..., description="Project name")
    deployment_type: DeploymentType = Field(..., description="Deployment type")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    description: str = Field(..., description="Project description")
    services: Dict[str, ServiceConfig] = Field(..., description="Service configurations")
    networks: List[NetworkConfig] = Field(
        default_factory=list,
        description="Network configurations"
    )
    global_environment: Dict[str, str] = Field(
        default_factory=dict,
        description="Global environment variables"
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Project labels"
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @validator('project_name')
    def validate_project_name(cls, v):
        """Validate project name"""
        if not v.isalnum() and not all(c in '-_' for c in v if not c.isalnum()):
            raise ValueError("Project name must be alphanumeric with only - and _")
        return v.lower()

class ProjectStatus(str, Enum):
    """Project build and deployment status"""
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"

class ServiceStatus(BaseModel):
    """Status information for a service"""
    status: str = Field(..., description="Current status")
    container_id: Optional[str] = Field(None, description="Container ID if running")
    health_status: Optional[str] = Field(None, description="Health check status")
    started_at: Optional[datetime] = Field(None, description="Start time")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    resource_usage: Optional[Dict] = Field(None, description="Resource usage statistics")

class ProjectStatus(BaseModel):
    """Complete project status"""
    project_name: str = Field(..., description="Project name")
    status: ProjectStatus = Field(..., description="Overall project status")
    services: Dict[str, ServiceStatus] = Field(..., description="Service statuses")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    build_logs: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Build logs by service"
    )