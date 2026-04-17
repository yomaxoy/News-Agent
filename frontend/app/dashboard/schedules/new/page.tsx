"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ScheduleForm from "@/components/ScheduleForm";
import api from "@/lib/api";

export default function NewSchedulePage() {
  const router = useRouter();
  const [sources, setSources] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchSources();
  }, []);

  const fetchSources = async () => {
    try {
      const res = await api.get("/api/sources");
      setSources(res.data);
    } catch (err) {
      console.error("Failed to load sources");
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return <div className="text-center py-6">Loading...</div>;
  }

  return (
    <div>
      <h1 className="text-3xl font-bold text-gray-900">Create Schedule</h1>
      <p className="mt-2 text-gray-600">Set up a new digest generation schedule</p>

      <div className="mt-8 max-w-2xl">
        <ScheduleForm
          sources={sources}
          onSuccess={() => router.push("/dashboard/schedules")}
        />
      </div>
    </div>
  );
}
