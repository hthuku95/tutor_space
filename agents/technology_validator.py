from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TechnologyValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the technology is valid")
    normalized_name: str = Field(description="The standardized name of the technology")
    category: str = Field(description="Primary category of the technology")
    subcategories: List[str] = Field(description="List of applicable subcategories")
    capabilities: List[str] = Field(description="Key capabilities and features")
    compatibility: List[str] = Field(description="Compatible technologies")
    validation_message: Optional[str] = Field(description="Message explaining validation result")

class TechnologyValidator:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=TechnologyValidationResult)

        self.validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technology expert who validates and classifies software development technologies.
            For each technology:
            1. Verify if it's a real technology used in software development
            2. Provide the standardized/normalized name
            3. Classify its primary category and subcategories
            4. List key capabilities and compatible technologies
            
            Be thorough in validation - only confirm real, established technologies.
            Provide specific validation messages for invalid technologies."""),
            ("human", """Analyze the following technology:
            Name: {name}
            Version: {version}
            Proposed Role: {role}
            
            Provide detailed validation results in the specified JSON format.""")
        ])

        self.validate_chain = self.validation_prompt | self.model | self.parser

    async def validate_technology(self, technology: Dict) -> TechnologyValidationResult:
        """Validate a single technology using the LLM"""
        try:
            result = await self.validate_chain.ainvoke({
                "name": technology["name"],
                "version": technology.get("version", ""),
                "role": technology["role"]
            })
            return result
        except Exception as e:
            logger.error(f"Error validating technology {technology['name']}: {str(e)}")
            raise

    async def validate_stack(self, technologies: List[Dict]) -> Dict:
        """Validate an entire technology stack and analyze compatibility"""
        validation_results = []
        validation_errors = []
        
        # Validate each technology
        for tech in technologies:
            try:
                result = await self.validate_technology(tech)
                if not result.is_valid:
                    validation_errors.append(result.validation_message)
                validation_results.append(result)
            except Exception as e:
                validation_errors.append(f"Error validating {tech['name']}: {str(e)}")

        # If any technologies are invalid, return errors
        if validation_errors:
            return {
                "is_valid": False,
                "errors": validation_errors
            }

        # Analyze stack compatibility
        stack_analysis = await self._analyze_stack_compatibility(validation_results)
        
        return {
            "is_valid": True,
            "validated_technologies": [
                {
                    "name": result.normalized_name,
                    "category": result.category,
                    "subcategories": result.subcategories,
                    "capabilities": result.capabilities
                }
                for result in validation_results
            ],
            "stack_analysis": stack_analysis
        }

    async def _analyze_stack_compatibility(
        self, 
        validated_techs: List[TechnologyValidationResult]
    ) -> Dict:
        """Analyze the compatibility and relationships between technologies"""
        stack_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a software architecture expert.
            Analyze the provided technology stack for:
            1. Overall architecture pattern identification
            2. Compatibility between components
            3. Potential integration challenges
            4. Recommended additional technologies or alternatives
            5. Best practices for this stack
            
            Provide detailed, actionable insights."""),
            ("human", """Analyze this technology stack:
            {stack_description}
            
            Provide analysis in JSON format with the following keys:
            - architecture_pattern
            - compatibility_matrix
            - integration_challenges
            - recommendations
            - best_practices""")
        ])

        stack_description = "\n".join([
            f"- {tech.normalized_name} ({tech.category}): {', '.join(tech.capabilities)}"
            for tech in validated_techs
        ])

        analysis_chain = stack_prompt | self.model | JsonOutputParser()
        
        try:
            return await analysis_chain.ainvoke({"stack_description": stack_description})
        except Exception as e:
            logger.error(f"Error analyzing stack compatibility: {str(e)}")
            raise