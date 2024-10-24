import os
import logging
from typing import Dict, List
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from ..types import (
    ProjectConfig,
    ServiceConfig,
    ServiceType,
    DeploymentType
)

logger = logging.getLogger(__name__)

class ServiceGenerator:
    """Generates service files and configurations for each service in a project"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.template_path = Path(__file__).parent / "templates"
        self.model = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = StrOutputParser()
        
        # Initialize Jinja2 environment for templates
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_path)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )

    async def generate_service(
        self,
        service_name: str,
        service_config: ServiceConfig,
        project_config: ProjectConfig
    ) -> Dict[str, str]:
        """
        Generate all files for a specific service
        
        Args:
            service_name: Name of the service
            service_config: Service configuration
            project_config: Complete project configuration
            
        Returns:
            Dictionary mapping file paths to their contents
        """
        logger.info(f"Generating service: {service_name}")
        
        # Create service directory
        service_path = self.base_path / service_name
        service_path.mkdir(parents=True, exist_ok=True)
        
        # Generate files based on service type
        files_dict = {}
        
        # Generate Dockerfile
        dockerfile_content = await self._generate_dockerfile(service_config)
        files_dict[str(service_path / "Dockerfile")] = dockerfile_content
        
        # Generate service-specific files
        type_specific_files = await self._generate_type_specific_files(
            service_name,
            service_config,
            project_config
        )
        files_dict.update(type_specific_files)
        
        # Generate configuration files
        config_files = await self._generate_config_files(
            service_name,
            service_config,
            project_config
        )
        files_dict.update(config_files)
        
        return files_dict

    async def _generate_dockerfile(self, service_config: ServiceConfig) -> str:
        """Generate Dockerfile for a service"""
        dockerfile_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Docker expert. Create a Dockerfile that follows best practices:
            - Use multi-stage builds when appropriate
            - Include security considerations
            - Optimize for build time and image size
            - Set up proper permissions
            - Configure healthchecks
            Do not include comments or explanations."""),
            ("human", """Generate a Dockerfile for:
            Base Image: {base_image}
            Service Type: {service_type}
            Environment: {environment}
            Ports: {ports}
            Command: {command}""")
        ])
        
        chain = dockerfile_prompt | self.model | self.parser
        
        dockerfile = await chain.ainvoke({
            "base_image": service_config.base_image,
            "service_type": service_config.type.value,
            "environment": service_config.environment,
            "ports": service_config.ports,
            "command": service_config.command
        })
        
        return dockerfile

    async def _generate_type_specific_files(
        self,
        service_name: str,
        service_config: ServiceConfig,
        project_config: ProjectConfig
    ) -> Dict[str, str]:
        """Generate files specific to service type"""
        generator_map = {
            ServiceType.FRONTEND: self._generate_frontend_files,
            ServiceType.BACKEND: self._generate_backend_files,
            ServiceType.DATABASE: self._generate_database_files,
            ServiceType.CACHE: self._generate_cache_files,
            ServiceType.MESSAGE_QUEUE: self._generate_queue_files,
        }
        
        generator = generator_map.get(service_config.type)
        if generator:
            return await generator(service_name, service_config, project_config)
        
        return {}

    async def _generate_frontend_files(
        self,
        service_name: str,
        service_config: ServiceConfig,
        project_config: ProjectConfig
    ) -> Dict[str, str]:
        """Generate frontend-specific files"""
        files = {}
        service_path = self.base_path / service_name
        
        # Generate package.json
        package_json = await self._generate_package_json(service_config)
        files[str(service_path / "package.json")] = package_json
        
        # Generate main application file
        app_content = await self._generate_frontend_app(service_config, project_config)
        files[str(service_path / "src" / "App.js")] = app_content
        
        # Generate component files
        components = await self._generate_frontend_components(service_config)
        for name, content in components.items():
            files[str(service_path / "src" / "components" / f"{name}.js")] = content
        
        # Generate configuration files
        files[str(service_path / ".env")] = self._generate_env_file(service_config)
        files[str(service_path / ".env.production")] = self._generate_env_file(
            service_config,
            DeploymentType.PRODUCTION
        )
        
        return files

    async def _generate_backend_files(
        self,
        service_name: str,
        service_config: ServiceConfig,
        project_config: ProjectConfig
    ) -> Dict[str, str]:
        """Generate backend-specific files"""
        files = {}
        service_path = self.base_path / service_name
        
        # Generate main application file
        app_content = await self._generate_backend_app(service_config, project_config)
        main_file = "app.py" if "python" in service_config.base_image.lower() else "index.js"
        files[str(service_path / main_file)] = app_content
        
        # Generate API routes
        api_routes = await self._generate_api_routes(service_config)
        api_dir = service_path / "routes"
        for name, content in api_routes.items():
            files[str(api_dir / name)] = content
        
        # Generate database models/schemas
        models = await self._generate_models(service_config)
        models_dir = service_path / "models"
        for name, content in models.items():
            files[str(models_dir / name)] = content
        
        # Generate configuration files
        files[str(service_path / "config.py")] = await self._generate_backend_config(
            service_config,
            project_config
        )
        
        return files

    async def _generate_database_files(
        self,
        service_name: str,
        service_config: ServiceConfig,
        project_config: ProjectConfig
    ) -> Dict[str, str]:
        """Generate database-specific files"""
        files = {}
        service_path = self.base_path / service_name
        
        # Generate initialization scripts
        init_scripts = await self._generate_db_init_scripts(service_config)
        scripts_dir = service_path / "init"
        for name, content in init_scripts.items():
            files[str(scripts_dir / name)] = content
        
        # Generate configuration files
        files[str(service_path / "my.cnf")] = await self._generate_db_config(service_config)
        
        return files

    async def _generate_config_files(
        self,
        service_name: str,
        service_config: ServiceConfig,
        project_config: ProjectConfig
    ) -> Dict[str, str]:
        """Generate common configuration files"""
        files = {}
        service_path = self.base_path / service_name
        
        # Generate environment variables
        files[str(service_path / ".env")] = self._generate_env_file(service_config)
        
        # Generate service-specific configs
        if service_config.type == ServiceType.FRONTEND:
            files[str(service_path / "nginx.conf")] = await self._generate_nginx_config(
                service_config
            )
        
        # Generate health check script if needed
        if service_config.health_check:
            files[str(service_path / "healthcheck.sh")] = self._generate_healthcheck(
                service_config.health_check
            )
        
        return files

    def _generate_env_file(
        self,
        service_config: ServiceConfig,
        deployment_type: DeploymentType = DeploymentType.DEVELOPMENT
    ) -> str:
        """Generate environment file content"""
        env_template = self.jinja_env.get_template(f"env/{deployment_type.value}.env.j2")
        return env_template.render(
            service_config=service_config,
            deployment_type=deployment_type
        )

    def _generate_healthcheck(self, health_check_config: Dict) -> str:
        """Generate health check script"""
        healthcheck_template = self.jinja_env.get_template("healthcheck.sh.j2")
        return healthcheck_template.render(config=health_check_config)