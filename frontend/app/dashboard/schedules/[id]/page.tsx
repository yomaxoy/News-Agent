"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import ScheduleForm from "@/components/ScheduleForm";
import api from "@/lib/api";

export default function EditSchedulePage() {
  const router = useRouter();
  const params = useParams();
  const [schedule, setSchedule] = useState<any>(null);
  const [sources, setSources] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, [params.id]);

  const fetchData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [scheduleRes, sourcesRes] = await Promise.all([
        api.get(`/api/schedules/${params.id}`),
        api.get("/api/sources"),
      ]);
      setSchedule(scheduleRes.data);
      setSources(sourcesRes.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to load data");
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return <div className="text-center py-6">Loading schedule...</div>;
  }

  if (error || !schedule) {
    return (
      <div>
        <div className="rounded-md bg-red-50 p-4">
          <p className="text-sm font-medium text-red-800">
            {error || "Schedule not found"}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-3xl font-bold text-gray-900">Edit Schedule</h1>
      <p className="mt-2 text-gray-600">Update digest schedule settings</p>

      <div className="mt-8 max-w-2xl">
        <ScheduleForm
          initialData={schedule}
          sources={sources}
          onSuccess={() => router.push("/dashboard/schedules")}
        />
      </div>
    </div>
  );
}
