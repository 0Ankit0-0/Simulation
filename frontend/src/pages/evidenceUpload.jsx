import { useState } from "react";
import api from "../services/api";

function EvidenceUpload() {
  const [file, setFile] = useState(null);
  const [msg, setMsg] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMsg("");
  };

  const handleUpload = async () => {
    if (!file) {
      setMsg("Please select a file to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await api.post("/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setMsg(res.data.message);
      setFile(null);
    } catch (err) {
      setMsg(err.response?.data?.error || "Upload failed");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-6">
          Upload Evidence
        </h1>
        <div className="mb-4">
          <input
            type="file"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
        </div>
        <button
          onClick={handleUpload}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition duration-300 ease-in-out"
        >
          Upload
        </button>
        {msg && (
          <p
            className={`mt-4 text-center ${
              msg.includes("failed") || msg.includes("select a file")
                ? "text-red-500"
                : "text-green-600"
            } font-medium`}
          >
            {msg}
          </p>
        )}

        <div className="mt-4 text-center">
          <p className="text-gray-700">
            {file ? `Selected file: ` : `No file selected`}
            <span className="font-semibold text-blue-800">
              {file ? file.name : ""}
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}

export default EvidenceUpload;
