from ai_agents.posecute import ProsecutorAgent
from ai_agents.defense import DefenseAgent
from ai_agents.judge import JudgeAgent

# Mock data for testing
case_data = {
    'case_id': 'TEST-001',
    'title': 'Test Case',
    'summary': 'A case involving theft and criminal activities',
    'description': 'Detailed case description for testing'
}

evidence = [
    {
        'filename': 'witness_statement.txt',
        'type': 'witness',
        'text': 'I saw the defendant commit the crime',
        'confidence': 0.9
    },
    {
        'filename': 'forensic_report.pdf',
        'type': 'forensic',
        'text': 'DNA evidence matches the defendant',
        'confidence': 0.95
    }
]

laws = {
    'ipc': {
        '379': {
            'title': 'Theft',
            'description': 'Punishment for theft',
            'keywords': ['theft', 'stolen', 'property']
        },
        '302': {
            'title': 'Murder',
            'description': 'Punishment for murder',
            'keywords': ['murder', 'death', 'kill']
        }
    }
}

def test_agents():
    # Initialize agents
    prosecutor = ProsecutorAgent(
        name="Test Prosecutor",
        case_data=case_data,
        evidence=evidence,
        laws=laws
    )
    
    defense = DefenseAgent(
        name="Test Defense",
        case_data=case_data,
        evidence=evidence,
        laws=laws
    )
    
    judge = JudgeAgent(
        name="Test Judge",
        case_data=case_data,
        evidence=evidence,
        laws=laws
    )
    
    # Test prosecutor
    print("=== PROSECUTOR TEST ===")
    prosecutor_thought = prosecutor.think()
    prosecutor_statement = prosecutor.speak(prosecutor_thought)
    print("Prosecutor Statement:", prosecutor_statement['statement'])
    print("Evidence Used:", prosecutor_statement.get('evidence'))
    print("Section Applied:", prosecutor_statement.get('section'))
    
    # Test defense
    print("\n=== DEFENSE TEST ===")
    defense_thought = defense.think(prosecutor_statement['statement'])
    defense_statement = defense.speak(defense_thought)
    print("Defense Statement:", defense_statement['statement'])
    print("Evidence Used:", defense_statement.get('evidence'))
    print("Section Applied:", defense_statement.get('section'))
    
    # Test judge
    print("\n=== JUDGE TEST ===")
    judge_thought = judge.think()
    judge_statement = judge.speak(judge_thought)
    print("Judge Statement:", judge_statement['statement'])
    
    # Test verdict
    final_arguments = [prosecutor_statement, defense_statement]
    verdict = judge.render_verdict(final_arguments)
    print("\n=== VERDICT ===")
    print("Verdict:", verdict['verdict'])
    print("Reasoning:", verdict['reasoning'])
    if verdict.get('sentence'):
        print("Sentence:", verdict['sentence'])

if __name__ == "__main__":
    test_agents()