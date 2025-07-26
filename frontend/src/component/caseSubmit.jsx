import React, { useState } from "react";
import api from "../services/api";
import { useNavigate } from "react-router-dom";

function CaseSubmit() {
  const [form, setForm] = useState({
    title: "",
    description: "",
    case_type: "criminal",
  });
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [response, setResponse] = useState("");
  const [error, setError] = useState("");

  const navigate = useNavigate(); 

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleFileChange = (e) => {
    setFiles([...e.target.files]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!form.title || !form.description || files.length === 0) {
      setError("All fields & evidence files are required.");
      return;
    }

    const formData = new FormData();
    formData.append("title", form.title);
    formData.append("description", form.description);
    formData.append("case_type", form.case_type);
    files.forEach((file) => formData.append("evidence", file));

    setUploading(true);
    setError("");
    setResponse("");

    try {
      const res = await api.post("/submit_case", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const caseId = res.data.case_id;
      setResponse(res.data.message + " | ID: " + caseId);
      setForm({ title: "", description: "", case_type: "criminal" });
      setFiles([]);

      // Navigate to review page with the actual case ID
      setTimeout(() => {
        navigate(`/review/${caseId}`);
      }, 1500); // Give user time to see the success message
    } catch (err) {
      console.error(err);
      const errorMessage =
        err.response?.data?.error || err.message || "Unknown error";
      setError("Error: " + errorMessage);
      setResponse("");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto p-6 bg-white shadow rounded">
      <h2 className="text-2xl font-bold mb-4">📝 Submit New Case</h2>

      <form onSubmit={handleSubmit}>
        <input
          name="title"
          value={form.title}
          onChange={handleChange}
          placeholder="Case Title"
          className="w-full p-2 mb-3 border rounded"
          required
        />

        <textarea
          name="description"
          value={form.description}
          onChange={handleChange}
          placeholder="Case Description"
          rows={4}
          className="w-full p-2 mb-3 border rounded"
          required
        />

        <select
          name="case_type"
          value={form.case_type}
          onChange={handleChange}
          className="w-full p-2 mb-3 border rounded"
        >
          <option value="criminal">Criminal</option>
          <option value="civil">Civil</option>
          <option value="constitutional">Constitutional</option>
          <option value="family">Family</option>
        </select>

        <input
          type="file"
          multiple
          onChange={handleFileChange}
          className="mb-3"
          accept=".pdf,.docx,.jpg,.jpeg,.png"
        />

        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
          disabled={uploading}
        >
          {uploading ? "Submitting..." : "Submit Case"}
        </button>
      </form>

      {response && <p className="mt-4 text-green-600">{response}</p>}
      {error && <p className="mt-4 text-red-600">{error}</p>}
    </div>
  );
}

export default CaseSubmit;
