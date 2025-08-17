import axios from "axios";

// Use the correct GitHub Codespaces backend URL
const API_BASE_URL =
  "https://curly-space-doodle-69wwpx655p96f5gjq-5000.app.github.dev/api" || "http://localhost:5000/api";

const api = axios.create({
  baseURL: API_BASE_URL,
  // timeout: 120000, // 120 second timeout for file uploads
  withCredentials: false, // GitHub Codespaces doesn't need credentials for public ports
  headers: {
    Accept: "application/json",
  },
});

// Request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log("Making request to:", config.baseURL + config.url);
    console.log("Method:", config.method);
    console.log("Headers:", config.headers);

    // Don't set Content-Type for FormData, let browser set it with boundary
    if (config.data instanceof FormData) {
      delete config.headers["Content-Type"];
    }

    return config;
  },
  (error) => {
    console.error("Request interceptor error:", error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log("Response received:", response.status, response.statusText);
    return response;
  },
  (error) => {
    console.error("API Error:", error);

    if (error.response) {
      // The request was made and the server responded with a status code
      console.error(
        "Error Response:",
        error.response.status,
        error.response.data
      );
    } else if (error.request) {
      // The request was made but no response was received
      console.error("No response received:", error.request);
    } else {
      // Something happened in setting up the request
      console.error("Error setting up request:", error.message);
    }

    return Promise.reject(error);
  }
);

export default api;
