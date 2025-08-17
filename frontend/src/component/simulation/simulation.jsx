import React, { useEffect, useState } from "react";
import api from "../../services/api"

export default function Simulation() {
  const [caseData, setCaseData] = useState(null);
  const [dialog, setDialog] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const caseId = window.location.pathname.split("/").pop();

  useEffect(() => {
    if (!caseId) {
      setError("No case ID found in URL.");
      setLoading(false);
      return;
    }

    api
      .get(`/api/get_case/${caseId}`)
      .then((res) => {
        setCaseData(res.data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching case data:", err);
        setError("Failed to load case data.");
        setLoading(false);
      });
  }, [caseId]);

  const startSimulation = () => {
    setLoading(true);
    setError(null);
    api
      .post(`/api/start_simulation/${caseId}`)
      .then((res) => {
        setDialog(res.data.dialog);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error starting simulation:", err);
        setError("Failed to start simulation.");
        setLoading(false);
      });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="text-xl font-semibold text-gray-700">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-red-100">
        <div className="text-xl font-semibold text-red-700 p-4 rounded-md border border-red-400">
          Error: {error}
        </div>
      </div>
    );
  }

  if (!caseData) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-yellow-100">
        <div className="text-xl font-semibold text-yellow-700">
          No case data available
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 bg-white shadow-lg rounded-lg my-8">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
        AI Courtroom Simulation
      </h2>

      <div className="bg-gray-50 p-4 rounded-md mb-6 border border-gray-200">
        <p className="mb-2">
          <strong className="text-gray-700">Case Title: </strong>{" "}
          <span className="text-gray-900">{caseData.title}</span>
        </p>
        <p>
          <strong className="text-gray-700">Parties: </strong>{" "}
          <span className="text-gray-900">
            {caseData.plaintiff} vs {caseData.defendant}
          </span>
        </p>
      </div>

      <button
        type="button"
        onClick={startSimulation}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 ease-in-out shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-75"
        disabled={loading}
      >
        {loading ? "Starting Simulation..." : "Start Simulation"}
      </button>

      {dialog.length > 0 && (
        <div className="mt-8 border-t pt-6 border-gray-200">
          <h3 className="text-2xl font-semibold text-gray-800 mb-4">
            Simulation Dialog
          </h3>
          <div className="space-y-4">
            {dialog.map((turn, index) => (
              <div
                key={index}
                className="p-4 bg-white rounded-lg shadow-sm border border-gray-100"
              >
                <p className="mb-1">
                  <strong className="text-indigo-600 capitalize">
                    {turn.role}:{" "}
                  </strong>{" "}
                  <span className="text-gray-800">{turn.statement}</span>
                </p>
                {turn.thought && (
                  <p className="text-sm text-gray-600 italic mt-1">
                    <strong className="text-gray-700">Thought: </strong>{" "}
                    {turn.thought}
                  </p>
                )}
                {turn.evidence && (
                  <a
                    href={turn.evidence}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-500 hover:text-blue-700 text-sm mt-2 inline-block"
                  >
                    View Evidence
                  </a>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
