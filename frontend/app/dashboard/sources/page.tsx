"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import SourceCard from "@/components/SourceCard";
import api from "@/lib/api";

export default function SourcesPage() {
  const [sources, setSources] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSources();
  }, []);

  const fetchSources = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const res = await api.get("/api/sources");
      setSources(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to load sources");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSourceDeleted = () => {
    fetchSources();
  };

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">RSS Sources</h1>
          <p className="mt-2 text-gray-600">
            Manage your RSS feeds and sources
          </p>
        </div>
        <Link
          href="/dashboard/sources/new"
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
        >
          + Add Source
        </Link>
      </div>

      {error && (
        <div className="mt-6 rounded-md bg-red-50 p-4">
          <p className="text-sm font-medium text-red-800">{error}</p>
        </div>
      )}

      {isLoading ? (
        <div className="mt-6 text-center">
          <p className="text-gray-600">Loading sources...</p>
        </div>
      ) : sources.length === 0 ? (
        <div className="mt-12 text-center">
          <p className="text-gray-600 mb-4">No sources yet</p>
          <Link
            href="/dashboard/sources/new"
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-blue-600 bg-blue-50 hover:bg-blue-100"
          >
            Create your first source
          </Link>
        </div>
      ) : (
        <div className="mt-6 space-y-4">
          {sources.map((source) => (
            <SourceCard
              key={source.id}
              source={source}
              onDelete={handleSourceDeleted}
            />
          ))}
        </div>
      )}
    </div>
  );
}
