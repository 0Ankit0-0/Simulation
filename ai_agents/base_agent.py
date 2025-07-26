import logging
from typing import Dict, List, Optional, Tuple, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all legal agents with AI integration capabilities."""
    
    def __init__(self, name: str, role: str, case_data: Dict, evidence: List[Dict], laws: Dict):
        # Input validation
        if not all([name, role, case_data, evidence, laws]):
            raise ValueError("All parameters (name, role, case_data, evidence, laws) are required")
        
        self.name = name
        self.role = role
        self.case_data = case_data
        self.evidence = evidence
        self.laws = laws
        self.memory = []
        
        # AI integration properties
        self.local_model = None
        self.pretrained_model = None
        self.api_client = None
        
        logger.info(f"Initialized {self.__class__.__name__} agent: {name} as {role}")

    def set_ai_models(self, local_model=None, pretrained_model=None, api_client=None):
        """Set AI models for fallback mechanism."""
        self.local_model = local_model
        self.pretrained_model = pretrained_model
        self.api_client = api_client

    def think(self, last_statement: Optional[str] = None) -> Tuple[str, Optional[str], Any]:
        """
        Plan what to say based on case facts, role, and last statement from other agent.
        Returns: (thought_process: str, target_evidence: str or None, additional_data: any)
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def speak(self, thought_result: Tuple[str, Optional[str], Any]) -> Dict[str, Any]:
        """Generate a final statement from the thought result.
        Returns: dict { 'role': str, 'statement': str, 'thought': str, 'evidence': optional filename }
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def remember(self, statement: str, thought: str) -> None:
        """Save to agents memory for future turn."""
        self.memory.append({"statement": statement, "thought": thought})

    def get_relevant_evidence(self, keywords: List[str]) -> List[Dict]:
        """Helper method to find relevant evidence based on keywords."""
        if not keywords:
            return []
        
        relevant_evidence = []
        for ev in self.evidence:
            if any(keyword.lower() in ev.get("text", "").lower() for keyword in keywords):
                relevant_evidence.append(ev)
        return relevant_evidence

    def _try_ai_generation(self, prompt: str, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Common AI generation method with fallback mechanism.
        Returns: dict with 'thought', 'evidence', and optional 'section' keys
        """
        
        # 1. Try local model first
        if self.local_model:
            try:
                logger.info(f"Attempting {method_name} with local model")
                result = getattr(self.local_model, method_name)(
                    self.case_data, prompt, self.evidence, self.laws
                )
                logger.info(f"Local model {method_name} successful")
                return result
            except Exception as e:
                logger.warning(f"Local model {method_name} failed: {e}")

        # 2. Try pretrained model
        if self.pretrained_model:
            try:
                logger.info(f"Attempting {method_name} with pretrained model")
                result = getattr(self.pretrained_model, method_name)(
                    self.case_data, prompt, self.evidence, self.laws
                )
                logger.info(f"Pretrained model {method_name} successful")
                return result
            except Exception as e:
                logger.warning(f"Pretrained model {method_name} failed: {e}")

        # 3. Try API clients
        if self.api_client:
            # Try Gemini API
            if "gemini" in self.api_client:
                try:
                    logger.info(f"Attempting {method_name} with Gemini API")
                    response = self.api_client["gemini"].generate(prompt)
                    logger.info(f"Gemini API {method_name} successful")
                    return {
                        "thought": response.get("thought", ""),
                        "evidence": response.get("evidence", None),
                        "section": response.get("section", None)
                    }
                except Exception as e:
                    logger.warning(f"Gemini API {method_name} failed: {e}")

            # Try OpenAI API
            openai_keys = ["chatgpt", "OpenAI", "openai"]
            for key in openai_keys:
                if key in self.api_client:
                    try:
                        logger.info(f"Attempting {method_name} with OpenAI API ({key})")
                        response = self.api_client[key].generate(prompt)
                        logger.info(f"OpenAI API {method_name} successful")
                        return {
                            "thought": response.get("thought", ""),
                            "evidence": response.get("evidence", None),
                            "section": response.get("section", None)
                        }
                    except Exception as e:
                        logger.warning(f"OpenAI API {method_name} failed: {e}")
                    break

        logger.error(f"All AI generation methods failed for {method_name}")
        return None

    def _create_base_prompt(self, last_statement: Optional[str] = None) -> str:
        """Create base prompt with case information."""
        prompt = f"""
        Case Summary: {self.case_data.get('summary', '')}
        Case Details: {self.case_data}
        Available Evidence: {self.evidence}
        Applicable Laws: {self.laws}
        """
        
        if last_statement:
            prompt += f"\nLast Statement from opposing counsel: {last_statement}"
        
        if self.memory:
            prompt += f"\nPrevious statements in this case: {self.memory[-3:]}"  # Last 3 for context
        
        return prompt