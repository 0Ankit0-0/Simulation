import random
from typing import Dict, List, Optional, Tuple, Any
from .base_agent import BaseAgent


class JudgeAgent(BaseAgent):
    """Judge agent that provides impartial judgment based on legal arguments and evidence."""

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
            name=name or "Justice Sharma",
            role=role or "judge",
            case_data=case_data,
            evidence=evidence,
            laws=laws,
        )

        # Set AI models for fallback mechanism
        self.set_ai_models(local_model, pretrained_model, api_client)

        # Judge-specific attributes
        self.prosecution_arguments = []
        self.defense_arguments = []
        self.verdict_rendered = False

    def think(
        self, last_statement: Optional[str] = None
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Analyze the case and provide judicial reasoning.
        Returns: (thought_process: str, referenced_evidence: str or None, applied_section: str or None)
        """

        # Create judge-specific prompt
        base_prompt = self._create_base_prompt(last_statement)
        judge_prompt = f"""
        {base_prompt}
        
        Role: You are an experienced and impartial judge presiding over this case.
        Task: Provide judicial analysis that:
        1. Evaluates the strength of both prosecution and defense arguments
        2. Analyzes the admissibility and weight of evidence presented
        3. Applies relevant legal principles and precedents
        4. Considers the burden of proof and standard of evidence
        5. Maintains complete impartiality and objectivity
        
        Prosecution Arguments: {self.prosecution_arguments}
        Defense Arguments: {self.defense_arguments}
        
        Please provide:
        - Detailed judicial reasoning and analysis
        - Assessment of key evidence (if any)
        - Applicable legal sections and their interpretation
        
        Focus on legal principles, evidence evaluation, and procedural fairness.
        Consider the presumption of innocence and burden of proof standards.
        """

        # Try AI generation with fallback
        ai_result = self._try_ai_generation(judge_prompt, "generate_judgment")

        if ai_result:
            return (
                ai_result.get("thought", ""),
                ai_result.get("evidence", None),
                ai_result.get("section", None),
            )

        # Fallback to rule-based judicial analysis
        return self._fallback_judicial_analysis(last_statement)

    def _fallback_judicial_analysis(
        self, last_statement: Optional[str] = None
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Fallback judicial analysis using rule-based approach."""

        # Analyze case complexity and evidence
        evidence_strength = self._evaluate_evidence_strength()
        legal_complexity = self._assess_legal_complexity()

        # Judicial reasoning
        thought = "The Court has carefully considered the arguments presented by both the prosecution and defense. "

        if evidence_strength > 0.7:
            thought += "The evidence presented appears to be substantial and credible. "
        elif evidence_strength > 0.4:
            thought += "The evidence presented requires careful examination and may be circumstantial. "
        else:
            thought += "The evidence presented appears to be limited and may not meet the required standard. "

        thought += "The Court notes that the burden of proof lies with the prosecution to establish guilt beyond reasonable doubt. "

        if self.prosecution_arguments and self.defense_arguments:
            thought += "Both sides have presented their arguments, and the Court must weigh them against the applicable law. "

        # Reference applicable legal sections
        referenced_sections = self._identify_applicable_sections()
        primary_section = referenced_sections[0] if referenced_sections else None

        if primary_section:
            section_info = self.laws.get("ipc", {}).get(primary_section, {})
            thought += f"The Court finds that IPC Section {primary_section} ({section_info.get('title', '')}) is particularly relevant to this case. "

        thought += "The Court will render its decision based on the law, evidence, and principles of natural justice."

        # Find most relevant evidence
        relevant_evidence = None
        if self.evidence:
            # Simple heuristic: pick evidence with most keywords matching case summary
            case_keywords = self.case_data.get("summary", "").lower().split()
            best_evidence = None
            best_score = 0

            for ev in self.evidence:
                score = sum(
                    1
                    for keyword in case_keywords
                    if keyword in ev.get("text", "").lower()
                )
                if score > best_score:
                    best_score = score
                    best_evidence = ev

            if best_evidence:
                relevant_evidence = best_evidence.get("filename")

        return thought, relevant_evidence, primary_section

    def _evaluate_evidence_strength(self) -> float:
        """Evaluate the overall strength of presented evidence."""
        if not self.evidence:
            return 0.0

        # Simple scoring based on evidence types and content
        score = 0
        for ev in self.evidence:
            ev_type = ev.get("type", "").lower()
            if ev_type in ["forensic", "dna", "fingerprint", "video", "audio"]:
                score += 0.3
            elif ev_type in ["witness", "testimony", "statement"]:
                score += 0.2
            elif ev_type in ["document", "record", "report"]:
                score += 0.1

            # Consider content quality (length as proxy)
            content_length = len(ev.get("text", ""))
            if content_length > 500:
                score += 0.1

        return min(score, 1.0)

    def _assess_legal_complexity(self) -> float:
        """Assess the legal complexity of the case."""
        complexity_score = 0

        # Multiple IPC sections indicate complexity
        case_summary = self.case_data.get("summary", "")
        section_count = sum(
            1
            for section in self.laws.get("ipc", {}).keys()
            if any(
                keyword in case_summary.lower()
                for keyword in self.laws["ipc"][section].get("keywords", [])
            )
        )

        complexity_score += min(section_count * 0.2, 0.6)

        # Multiple evidence types indicate complexity
        evidence_types = set(ev.get("type", "") for ev in self.evidence)
        complexity_score += min(len(evidence_types) * 0.1, 0.4)

        return min(complexity_score, 1.0)

    def _identify_applicable_sections(self) -> List[str]:
        """Identify the most applicable IPC sections for the case."""
        case_summary = self.case_data.get("summary", "").lower()
        applicable_sections = []

        for section_num, section_data in self.laws.get("ipc", {}).items():
            keywords = section_data.get("keywords", [])
            if any(keyword.lower() in case_summary for keyword in keywords):
                applicable_sections.append(section_num)

        return applicable_sections

    def speak(
        self, thought_result: Tuple[str, Optional[str], Optional[str]]
    ) -> Dict[str, Any]:
        """Generate judicial statement from thought result."""
        thought, evidence_file, section = thought_result

        # Formal judicial statement
        statement = "The Court has given due consideration to all arguments and evidence presented by both parties. "
        statement += thought

        # Reference to evidence if applicable
        if evidence_file:
            statement += f" The Court particularly notes the significance of evidence file {evidence_file} in this matter. "

        # Legal section reference
        if section:
            section_info = self.laws.get("ipc", {}).get(section, {})
            statement += f" The Court finds that IPC Section {section} ({section_info.get('title', '')}) is central to the legal determination in this case. "

        # Procedural note
        statement += "The Court ensures that all proceedings are conducted in accordance with the principles of natural justice and due process. "

        return {
            "role": self.role,
            "statement": statement,
            "thought": thought,
            "evidence": evidence_file,
            "section": section,
        }

    def record_argument(self, agent_role: str, statement: str, thought: str) -> None:
        """Record arguments from prosecution and defense for later analysis."""
        argument_record = {
            "statement": statement,
            "thought": thought,
            "timestamp": len(self.memory) + 1,
        }

        if agent_role == "prosecutor":
            self.prosecution_arguments.append(argument_record)
        elif agent_role in ["defendant", "defense"]:
            self.defense_arguments.append(argument_record)

    def render_verdict(self, final_arguments: List[Dict]) -> Dict[str, Any]:
        """
        Render final verdict based on all arguments and evidence.
        Returns: dict with verdict, reasoning, and sentence
        """
        if self.verdict_rendered:
            return {"error": "Verdict already rendered"}

        # Create verdict prompt
        verdict_prompt = f"""
        {self._create_base_prompt()}
        
        Role: You are rendering the final verdict in this case.
        
        All Prosecution Arguments: {self.prosecution_arguments}
        All Defense Arguments: {self.defense_arguments}
        Final Arguments: {final_arguments}
        
        Task: Render a fair and impartial verdict that:
        1. Determines guilt or innocence based on evidence
        2. Applies the correct legal standard (beyond reasonable doubt)
        3. Considers all arguments presented
        4. Provides clear reasoning for the decision
        5. Determines appropriate sentence if guilty
        
        Please provide:
        - Clear verdict (Guilty/Not Guilty)
        - Detailed reasoning for the verdict
        - Appropriate sentence if applicable
        - Legal sections that apply to the sentence
        """

        # Try AI generation for verdict
        ai_result = self._try_ai_generation(verdict_prompt, "generate_verdict")

        if ai_result:
            verdict_data = {
                "verdict": ai_result.get("verdict", "Not Guilty"),
                "reasoning": ai_result.get("reasoning", ""),
                "sentence": ai_result.get("sentence", None),
                "applied_sections": ai_result.get("sections", []),
            }
        else:
            # Fallback verdict logic
            verdict_data = self._fallback_verdict()

        self.verdict_rendered = True
        return verdict_data

    def _fallback_verdict(self) -> Dict[str, Any]:
        """Fallback verdict generation using rule-based approach."""

        # Simple scoring system
        prosecution_score = len(self.prosecution_arguments) * 0.3
        defense_score = len(self.defense_arguments) * 0.3
        evidence_score = self._evaluate_evidence_strength() * 0.4

        # Adjust for evidence quality
        total_score = prosecution_score + evidence_score - defense_score

        # Determine verdict (higher threshold for conviction)
        if total_score > 0.7:
            verdict = "Guilty"
            reasoning = "The Court finds that the prosecution has proven its case beyond reasonable doubt. The evidence presented is credible and substantial."

            # Determine sentence based on applicable sections
            applicable_sections = self._identify_applicable_sections()
            if applicable_sections:
                primary_section = applicable_sections[0]
                section_info = self.laws.get("ipc", {}).get(primary_section, {})
                sentence = f"The accused is sentenced according to IPC Section {primary_section}."
            else:
                sentence = "The Court will determine appropriate sentencing based on the gravity of the offense."
        else:
            verdict = "Not Guilty"
            reasoning = "The Court finds that the prosecution has not proven its case beyond reasonable doubt. The evidence is insufficient to establish guilt."
            sentence = None
            applicable_sections = []

        return {
            "verdict": verdict,
            "reasoning": reasoning,
            "sentence": sentence,
            "applied_sections": applicable_sections,
        }
