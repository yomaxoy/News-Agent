"use client";

import Link from "next/link";
import { useState } from "react";
import api from "@/lib/api";

interface ScheduleCardProps {
  schedule: any;
  onDelete: () => void;
}

export default function ScheduleCard({ schedule, onDelete }: ScheduleCardProps) {
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDelete = async () => {
    if (!confirm("Are you sure you want to delete this schedule?")) return;

    setIsDeleting(true);
    try {
      await api.delete(`/api/schedules/${schedule.id}`);
      onDelete();
    } catch (err) {
      alert("Failed to delete schedule");
    } finally {
      setIsDeleting(false);
    }
  };

  const nextRun = schedule.next_run_at
    ? new Date(schedule.next_run_at).toLocaleString()
    : "Not scheduled";

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h3 className="text-lg font-medium text-gray-900">{schedule.name}</h3>
          <p className="mt-1 text-sm text-gray-500">
            Cron: <code className="bg-gray-100 px-2 py-1 rounded">{schedule.cron_expression}</code>
          </p>
          <p className="mt-1 text-sm text-gray-500">
            Timezone: {schedule.timezone}
          </p>
          <p className="mt-1 text-sm text-gray-500">
            Next run: {nextRun}
          </p>
          <div className="mt-3 flex items-center space-x-2">
            {schedule.is_active ? (
              <span className="inline-block px-2 py-1 text-xs font-medium rounded-full bg-green-100 text-green-800">
                Active
              </span>
            ) : (
              <span className="inline-block px-2 py-1 text-xs font-medium rounded-full bg-gray-100 text-gray-800">
                Inactive
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-2 ml-4">
          <Link
            href={`/dashboard/schedules/${schedule.id}`}
            className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            Edit
          </Link>
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-red-600 hover:bg-red-700 disabled:opacity-50"
          >
            {isDeleting ? "..." : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}
