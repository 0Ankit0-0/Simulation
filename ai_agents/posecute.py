import random
from typing import Dict, List, Optional, Tuple, Any
from .base_agent import BaseAgent


class ProsecutorAgent(BaseAgent):
    """Prosecutor agent with AI-powered case building and legal strategy."""

    def __init__(
        self,
        name: Optional[str] = None,
        role: Optional[str] = None,
        case_data: Dict = None,
        evidence: List[Dict] = None,
        laws: Dict = None,
        local_model=None,
        pretrained_model=None,
        api_client=None,
    ):
        super().__init__(
            name=name or "Ramesh",
            role=role or "prosecutor",
            case_data=case_data,
            evidence=evidence,
            laws=laws,
        )

        # Set AI models for fallback mechanism
        self.set_ai_models(local_model, pretrained_model, api_client)

    def think(
        self, last_statement: Optional[str] = None
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Plan prosecution strategy based on case facts and applicable laws.
        Returns: (thought_process: str, target_evidence: str or None, section: str)
        """

        # Create prosecution-specific prompt
        base_prompt = self._create_base_prompt(last_statement)
        prosecution_prompt = f"""
        {base_prompt}
        
        Role: You are an experienced prosecutor representing the state.
        Task: Build a strong prosecution case that:
        1. Identifies the most applicable criminal charges based on the facts
        2. Selects the strongest evidence to support the charges
        3. Anticipates and addresses potential defense arguments
        4. Applies relevant legal sections from the Indian Penal Code
        5. Responds to any defense statements made previously
        
        Please provide:
        - A detailed thought process for the prosecution strategy
        - Specific evidence file that best supports the charges
        - Most applicable IPC section for the charges
        
        Focus on establishing mens rea (criminal intent) and actus reus (criminal act).
        """

        # Try AI generation with fallback
        ai_result = self._try_ai_generation(prosecution_prompt, "generate_prosecution")

        if ai_result:
            return (
                ai_result.get("thought", ""),
                ai_result.get("evidence", None),
                ai_result.get("section", None),
            )

        # Fallback to rule-based prosecution if AI fails
        return self._fallback_prosecution_strategy(last_statement)

    def _fallback_prosecution_strategy(
        self, last_statement: Optional[str] = None
    ) -> Tuple[str, Optional[str], str]:
        """Fallback prosecution strategy using rule-based approach."""

        ipc_sections = self.laws.get("ipc", {})
        case_summary = self.case_data.get("summary", "").lower()

        # Find applicable IPC sections based on case facts
        possible_sections = []
        for section_num, section_data in ipc_sections.items():
            section_keywords = section_data.get("keywords", [])
            if any(keyword.lower() in case_summary for keyword in section_keywords):
                possible_sections.append((section_num, section_data))

        # If no specific sections found, use common criminal sections
        if not possible_sections:
            common_sections = ["302", "304", "323", "379", "420", "506"]
            for section_num in common_sections:
                if section_num in ipc_sections:
                    possible_sections.append((section_num, ipc_sections[section_num]))

        # Select the most serious applicable section
        selected_section = (
            possible_sections[0]
            if possible_sections
            else (
                "302",
                ipc_sections.get(
                    "302",
                    {
                        "title": "Murder",
                        "description": "Punishment for murder",
                        "keywords": ["murder", "death", "kill"],
                    },
                ),
            )
        )

        # Find supporting evidence
        keywords = selected_section[1].get("keywords", [])
        relevant_evidence = self.get_relevant_evidence(keywords)
        selected_evidence = (
            relevant_evidence[0]["filename"] if relevant_evidence else None
        )

        # Generate prosecution thought process
        thought = f"The state submits that the evidence clearly establishes that the accused has committed an offense under IPC Section {selected_section[0]}. "
        thought += f"The nature of the act constitutes {selected_section[1].get('title', 'a criminal offense')}. "

        if selected_evidence:
            thought += f"The evidence file {selected_evidence} provides crucial support for the prosecution's case. "

        thought += "The prosecution has established both the actus reus (criminal act) and mens rea (criminal intent) required for conviction. "

        if last_statement:
            thought += "In response to the defense's arguments, the prosecution maintains that the evidence is overwhelming and the charges are well-founded. "

        thought += "The state respectfully requests the court to find the accused guilty as charged."

        return thought, selected_evidence, selected_section[0]

    def speak(self, thought_result: Tuple[str, Optional[str], str]) -> Dict[str, Any]:
        """Generate prosecution statement from thought result."""
        thought, evidence_file, section = thought_result

        # Get section information
        section_info = self.laws.get("ipc", {}).get(section, {})
        section_title = section_info.get("title", "a criminal offense")

        # Build formal prosecution statement
        statement = f"Your Honor, the State respectfully submits that the evidence in this case clearly establishes that the accused has committed an offense under IPC Section {section}. "
        statement += f"The nature of the act constitutes {section_title}. "

        # Add evidence presentation
        if evidence_file:
            statement += f"The prosecution would like to present evidence file {evidence_file} which conclusively supports the charges against the accused. "

        # Add legal reasoning
        statement += "The prosecution has established beyond reasonable doubt both the criminal act and the requisite criminal intent. "

        # Add case-specific details
        if section_info.get("description"):
            statement += f"According to {section_info['description']}, the accused's actions fall squarely within the ambit of this section. "

        # Closing argument
        statement += "Based on the overwhelming evidence and the applicable law, the prosecution respectfully requests this honorable court to find the accused guilty as charged."

        return {
            "role": self.role,
            "statement": statement,
            "thought": thought,
            "evidence": evidence_file,
            "section": section,
        }
