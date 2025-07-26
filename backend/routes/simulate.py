from flask import Blueprint, request, jsonify
from ai_agents.courtroom import run_courtroom_simulation

simulate_bp = Blueprint("simulate", __name__)


@simulate_bp.route("/simulate", methods=["POST"])
def simulate():
    """
    Run a courtroom simulation with the provided case data, evidence, and laws.
    Expects JSON input with 'case_data', 'evidence', and 'laws'.
    """
    try:
        data = request.json
        case_data = data.get("case_data")
        evidence = data.get("evidence", [])
        laws = data.get("laws", {})

        if not case_data or not evidence or not laws:
            return jsonify({"error": "Missing required fields: case_data, evidence, or laws"}), 400

        # Run the courtroom simulation
        result = run_courtroom_simulation(case_data, evidence, laws)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500