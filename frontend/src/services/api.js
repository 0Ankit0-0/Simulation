import axios from "axios";

const API_BASE_URL = "https://solid-invention-r4wwx5966wqpfx759-5000.app.github.dev/api" || "http://127.0.0.1:5000/api" 
|| "https://solid-invention-r4wwx5966wqpfx759-5000.app.github.dev/api";

const api = axios.create({
  baseURL: API_BASE_URL,
});

export default api;
