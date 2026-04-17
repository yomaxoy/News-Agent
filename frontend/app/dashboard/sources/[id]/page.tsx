"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import SourceForm from "@/components/SourceForm";
import api from "@/lib/api";

export default function EditSourcePage() {
  const router = useRouter();
  const params = useParams();
  const [source, setSource] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSource();
  }, [params.id]);

  const fetchSource = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const res = await api.get(`/api/sources/${params.id}`);
      setSource(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to load source");
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return <div className="text-center py-6">Loading source...</div>;
  }

  if (error || !source) {
    return (
      <div>
        <div className="rounded-md bg-red-50 p-4">
          <p className="text-sm font-medium text-red-800">
            {error || "Source not found"}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-3xl font-bold text-gray-900">Edit Source</h1>
      <p className="mt-2 text-gray-600">Update RSS source details</p>

      <div className="mt-8 max-w-2xl">
        <SourceForm
          initialData={source}
          onSuccess={() => router.push("/dashboard/sources")}
        />
      </div>
    </div>
  );
}
