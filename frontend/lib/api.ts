import axios, { AxiosError } from "axios";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/+$/, "");

const api = axios.create({
  baseURL: API_URL,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true,
});

api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Redirect to login on unauthorized
      window.location.href = "/auth/login";
    }
    return Promise.reject(error);
  }
);

export default api;
