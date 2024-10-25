
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class MessageType(str, Enum):
    """Types of messages that can be exchanged between agents"""
    SERVICE_SPEC = "service_specification"
    CODE_REQUEST = "code_generation_request"
    CODE_RESPONSE = "code_generation_response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    CONFIG_REQUEST = "configuration_request"
    CONFIG_RESPONSE = "configuration_response"

class CodeGenerationPriority(str, Enum):
    """Priority levels for code generation requests"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ServiceRequest(BaseModel):
    """Request from ServiceGenerator to Agent 4"""
    request_id: str = Field(..., description="Unique identifier for this request")
    message_type: MessageType
    priority: CodeGenerationPriority
    service_spec: Dict[str, Any] = Field(..., description="Service specification")
    requirements: Dict[str, Any] = Field(..., description="Additional requirements")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    class Config:
        extra = "allow"

class CodeFile(BaseModel):
    """Represents a generated code file"""
    path: str = Field(..., description="Path where file should be created")
    content: str = Field(..., description="Content of the file")
    language: str = Field(..., description="Programming language")
    purpose: str = Field(..., description="Purpose of this file")

class ServiceResponse(BaseModel):
    """Response from Agent 4 to ServiceGenerator"""
    request_id: str = Field(..., description="ID of the original request")
    message_type: MessageType
    status: str = Field(..., description="Status of the code generation")
    files: List[CodeFile] = Field(..., description="Generated code files")
    additional_info: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional information about the generation"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings during generation"
    )

class AgentProtocol:
    """
    Protocol for communication between ServiceGenerator and Agent 4.
    Handles message formatting, validation, and routing.
    """

    def __init__(self):
        self.pending_requests: Dict[str, ServiceRequest] = {}
        self.responses: Dict[str, ServiceResponse] = {}

    def create_service_request(
        self,
        service_spec: Dict[str, Any],
        priority: CodeGenerationPriority = CodeGenerationPriority.MEDIUM,
        **kwargs
    ) -> ServiceRequest:
        """Create a new service request"""
        from uuid import uuid4
        
        request = ServiceRequest(
            request_id=str(uuid4()),
            message_type=MessageType.SERVICE_SPEC,
            priority=priority,
            service_spec=service_spec,
            requirements=kwargs.get('requirements', {}),
            context=kwargs.get('context', {})
        )
        
        self.pending_requests[request.request_id] = request
        return request

    def create_service_response(
        self,
        request_id: str,
        status: str,
        files: List[CodeFile],
        **kwargs
    ) -> ServiceResponse:
        """Create a new service response"""
        if request_id not in self.pending_requests:
            raise ValueError(f"Unknown request ID: {request_id}")
            
        response = ServiceResponse(
            request_id=request_id,
            message_type=MessageType.CODE_RESPONSE,
            status=status,
            files=files,
            additional_info=kwargs.get('additional_info', {}),
            warnings=kwargs.get('warnings', [])
        )
        
        self.responses[request_id] = response
        return response

    def validate_request(self, request: ServiceRequest) -> bool:
        """Validate a service request"""
        try:
            # Validate service specification structure
            required_fields = {'service_name', 'service_type', 'technology_stack'}
            if not all(field in request.service_spec for field in required_fields):
                return False
                
            # Validate technology stack
            tech_stack = request.service_spec['technology_stack']
            if not all(field in tech_stack for field in ['language', 'framework']):
                return False
                
            # Validate requirements
            if 'requirements' in request.service_spec:
                if not isinstance(request.service_spec['requirements'], dict):
                    return False
                    
            return True
            
        except Exception:
            return False

    def validate_response(self, response: ServiceResponse) -> bool:
        """Validate a service response"""
        try:
            # Check if request exists
            if response.request_id not in self.pending_requests:
                return False
                
            # Validate files
            if not response.files:
                return False
                
            for file in response.files:
                if not all(hasattr(file, field) 
                          for field in ['path', 'content', 'language']):
                    return False
                    
            return True
            
        except Exception:
            return False
