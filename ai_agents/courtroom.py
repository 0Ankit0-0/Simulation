from .posecute import ProsecutorAgent
from .defense import DefenseAgent
from .judge import JudgeAgent
import random

def run_courtroom_simulation():
    # Initialize agents
    prosecutor = ProsecutorAgent(case_data=case_data, evidence=evidence, laws=laws, name="Ramesh", role="prosecutor")
    defense = DefenseAgent(case_data=case_data, evidence=evidence, laws=laws, name="Suresh", role="defendant")
    judge = JudgeAgent(case_data=case_data, evidence=evidence, laws=laws, name="Justice Verma", role="judge")

    turns = []
    last_statement = None
    for _ in range(random.randint(10, 20)):  
        # Prosecutor's turn
        prosecution_thought = prosecutor.think(last_statement)
        prosecution_statement = prosecutor.speak(prosecution_thought)
        prosecutor.remember(prosecution_statement['statement'], prosecution_thought[0])
        turns.append(prosecution_statement)

        # Defense's turn
        defense_thought = defense.think(last_statement)
        defense_statement = defense.speak(defense_thought)
        defense.remember(defense_statement['statement'], defense_thought[0])
        turns.append(defense_statement)

        # Judge's turn
        judge_thought = judge.think(prosecution_statement['statement'], defense_statement['statement'])
        judge_statement = judge.speak(judge_thought)
        judge.remember(judge_statement['statement'], judge_thought[0])
        turns.append(judge_statement)

        last_statement = judge_statement['statement']  # Update last statement for next turn
        verdict = judge.get_verdict()  # Get verdict after each round
  
    return {
        'arguments': turns,
        "verdict": verdict if verdict else "No verdict yet"
    }