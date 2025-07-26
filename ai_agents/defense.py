import random
from typing import Dict, List, Optional, Tuple, Any
from ai_agents.base_agent import BaseAgent


class DefenseAgent(BaseAgent):
    """Defense attorney agent with AI-powered legal strategy generation."""

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
            name=name or "Suresh",
            role=role or "defendant",  
            case_data=case_data,
            evidence=evidence,
            laws=laws,
        )

        # Set AI models
        self.set_ai_models(local_model, pretrained_model, api_client)

    def think(
        self, last_statement: Optional[str] = None
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Plan defense strategy by refuting charges using counter evidence and applicable legal sections.
        Returns: (thought_process: str, target_evidence: str or None, section: str or None)
        """

        # Create defense-specific prompt
        base_prompt = self._create_base_prompt(last_statement)
        defense_prompt = f"""
        {base_prompt}
        
        Role: You are an experienced defense attorney.
        Task: Generate a defense strategy that:
        1. Refutes or challenges the prosecution's charges
        2. Identifies weaknesses in the prosecution's case
        3. Suggests applicable legal sections that favor the defense
        4. Recommends relevant evidence to support the defense
        5. Addresses any previous statements made by the prosecution
        
        Please provide:
        - A detailed thought process for the defense strategy
        - Specific evidence file that supports the defense (if available)
        - Applicable legal section that favors the defendant
        
        Focus on constitutional rights, burden of proof, and procedural safeguards.
        """

        # Try AI generation with fallback
        ai_result = self._try_ai_generation(defense_prompt, "generate_defense")

        if ai_result:
            return (
                ai_result.get("thought", ""),
                ai_result.get("evidence", None),
                ai_result.get("section", None),
            )

        # Fallback to rule-based defense if AI fails
        return self._fallback_defense_strategy(last_statement)

    def _fallback_defense_strategy(
        self, last_statement: Optional[str] = None
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Fallback defense strategy using rule-based approach."""

        # Look for defensive legal sections
        defensive_sections = {
            "76": {
                "title": "Act done by a person bound by law",
                "keywords": ["duty", "bound", "law"],
            },
            "79": {
                "title": "Act done by a person justified by law",
                "keywords": ["justified", "legal"],
            },
            "80": {
                "title": "Accident in doing a lawful act",
                "keywords": ["accident", "lawful"],
            },
            "81": {
                "title": "Act likely to cause harm but done without criminal intent",
                "keywords": ["intent", "harm"],
            },
            "84": {
                "title": "Act of a person of unsound mind",
                "keywords": ["mental", "mind", "unsound"],
            },
            "85": {
                "title": "Act of a person incapable of judgment",
                "keywords": ["incapable", "judgment"],
            },
        }

        # Check case summary for defensive keywords
        case_summary = self.case_data.get("summary", "").lower()
        possible_defenses = []

        for section_num, section_data in defensive_sections.items():
            if any(keyword in case_summary for keyword in section_data["keywords"]):
                possible_defenses.append((section_num, section_data))

        selected_defense = None
        if possible_defenses:
            selected_defense = random.choice(possible_defenses)

        # Look for supporting evidence
        evidence_keywords = ["witness", "alibi", "character", "medical", "expert"]
        relevant_evidence = self.get_relevant_evidence(evidence_keywords)
        selected_evidence = (
            relevant_evidence[0]["filename"] if relevant_evidence else None
        )

        # Generate thought process
        thought = "As defense counsel, I challenge the prosecution's case on several grounds. "

        if selected_defense:
            thought += f"The circumstances suggest that IPC Section {selected_defense[0]} ({selected_defense[1]['title']}) may apply, "
            thought += "which could provide a valid defense to the charges. "

        thought += "I emphasize that the burden of proof lies with the prosecution to establish guilt beyond reasonable doubt. "

        if last_statement:
            thought += "In response to the prosecution's arguments, I maintain that the evidence is insufficient and circumstantial. "

        thought += "The defense reserves the right to present additional evidence and witnesses to establish my client's innocence."

        section = selected_defense[0] if selected_defense else None

        return thought, selected_evidence, section

    def speak(
        self, thought_result: Tuple[str, Optional[str], Optional[str]]
    ) -> Dict[str, Any]:
        """Generate defense statement from thought result."""
        thought, evidence_file, section = thought_result

        # Start with respectful address to the court
        statement = f"Your Honor, I respectfully submit that the defense challenges the prosecution's case. {thought}"

        # Add legal section reference if available
        if section and section in self.laws.get("ipc", {}):
            section_info = self.laws["ipc"][section]
            statement += f" This matter may fall under IPC Section {section}: {section_info.get('title', '')}."

        # Add evidence submission if available
        if evidence_file:
            statement += f" I would like to submit the following evidence in support of the defense: {evidence_file}."

        # Add closing statement
        statement += " I urge the court to consider the constitutional principle of presumption of innocence and the high standard of proof required for conviction."

        return {
            "role": self.role,
            "statement": statement,
            "thought": thought,
            "evidence": evidence_file,
            "section": section,
        }
