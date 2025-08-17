import React, { useEffect, useState } from 'react';
import api from '../../services/api';
import { useParams, useNavigate } from 'react-router-dom';

const CaseReview = () => {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const [caseData, setCaseData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const fetchCase = async () => {
      try {
        const res = await api.get(`/get_case/${caseId}`);
        setCaseData(res.data);
        setError("");
      } catch (err) {
        console.error(err);
        setError("Failed to load case data. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    if (caseId) {
      fetchCase();
    }
  }, [caseId]);

  const handleApproval = (index, status) => {
    const updatedEvidence = [...caseData.evidence];
    updatedEvidence[index].approved = status;
    setCaseData({ ...caseData, evidence: updatedEvidence });
  };

  const handleTextChange = (index, newText) => {
    const updatedEvidence = [...caseData.evidence];
    updatedEvidence[index].text = newText;
    setCaseData({ ...caseData, evidence: updatedEvidence });
  };

  const handleSubmitReview = async () => {
    setSubmitting(true);
    try {
      await api.post('/review_case', {
        case_id: caseId,
        evidence: caseData.evidence,
      });
      navigate(`/simulate/${caseId}`);
    } catch (err) {
      console.error(err);
      setError("Failed to submit review. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) return <div className="p-4 text-white">Loading case...</div>;
  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!caseData) return <div className="p-4 text-white">No case data found.</div>;

  return (
    <div className="p-6 space-y-6 text-white min-h-screen bg-gray-900">
      <div className="mb-6">
        <button
          onClick={() => navigate('/')}
          className="bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded text-white mb-4"
        >
          ← Back to Home
        </button>
        <h1 className="text-3xl font-bold">🕵️ Case Review: {caseData.title}</h1>
        <p className="italic text-gray-300 mt-2">{caseData.description}</p>
        <div className="text-sm text-gray-400 mt-2">
          <span>Case ID: {caseData.case_id}</span> | 
          <span> Type: {caseData.case_type}</span> | 
          <span> Submitted: {new Date(caseData.submitted_at).toLocaleString()}</span>
        </div>
      </div>

      {caseData.evidence && caseData.evidence.length > 0 ? (
        caseData.evidence.map((item, i) => (
          <div key={i} className="bg-zinc-800 p-4 rounded-xl shadow-xl">
            <h3 className="text-xl font-semibold mb-2">{item.filename}</h3>
            <p className="mt-2 text-sm text-gray-400">
              <strong>Summary:</strong> {item.summary || "No summary available"}
            </p>
            <label className="block mt-3 text-sm font-medium text-gray-300">
              Extracted Text:
            </label>
            <textarea
              className="w-full mt-2 p-3 rounded bg-zinc-700 text-white border border-zinc-600 focus:border-blue-500 focus:outline-none"
              value={item.text || ""}
              onChange={(e) => handleTextChange(i, e.target.value)}
              rows={6}
              placeholder="No text extracted from this evidence..."
            />
            <div className="mt-4 flex gap-4">
              <button
                className={`px-4 py-2 rounded transition-colors ${
                  item.approved === true 
                    ? 'bg-green-600 hover:bg-green-700' 
                    : 'bg-green-800 hover:bg-green-700'
                }`}
                onClick={() => handleApproval(i, true)}
              >
                👍 Accept
              </button>
              <button
                className={`px-4 py-2 rounded transition-colors ${
                  item.approved === false 
                    ? 'bg-red-600 hover:bg-red-700' 
                    : 'bg-red-800 hover:bg-red-700'
                }`}
                onClick={() => handleApproval(i, false)}
              >
                👎 Reject
              </button>
              <span className="px-4 py-2 text-sm text-gray-400">
                Status: {item.approved === true ? 'Accepted' : item.approved === false ? 'Rejected' : 'Pending'}
              </span>
            </div>
          </div>
        ))
      ) : (
        <div className="bg-zinc-800 p-4 rounded-xl text-center text-gray-400">
          No evidence files found for this case.
        </div>
      )}

      <div className="flex gap-4 pt-4">
        <button
          onClick={handleSubmitReview}
          disabled={submitting}
          className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed px-6 py-3 rounded-xl text-white font-semibold transition-colors"
        >
          {submitting ? "Submitting..." : "✅ Start Simulation"}
        </button>
        <button
          onClick={() => navigate('/')}
          className="bg-gray-600 hover:bg-gray-700 px-6 py-3 rounded-xl text-white font-semibold transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

export default CaseReview;