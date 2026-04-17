"use client";

import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { scheduleSchema, ScheduleFormData } from "@/lib/schemas";
import api from "@/lib/api";

const CRON_PRESETS = [
  { label: "Every Day at 6 AM", value: "0 6 * * *" },
  { label: "Every Day at 12 PM", value: "0 12 * * *" },
  { label: "Every Day at 6 PM", value: "0 18 * * *" },
  { label: "Every Monday at 6 AM", value: "0 6 * * 1" },
  { label: "Every Week on Sunday", value: "0 6 * * 0" },
];

interface ScheduleFormProps {
  initialData?: any;
  sources: any[];
  onSuccess: () => void;
}

export default function ScheduleForm({
  initialData,
  sources,
  onSuccess,
}: ScheduleFormProps) {
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedSources, setSelectedSources] = useState<number[]>(
    initialData?.source_ids || []
  );

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm<ScheduleFormData>({
    resolver: zodResolver(scheduleSchema),
    defaultValues: initialData || {
      cron_expression: "0 6 * * *",
      timezone: "UTC",
      max_articles: 7,
    },
  });

  const cronValue = watch("cron_expression");

  const handleSourceToggle = (sourceId: number) => {
    setSelectedSources((prev) =>
      prev.includes(sourceId)
        ? prev.filter((id) => id !== sourceId)
        : [...prev, sourceId]
    );
  };

  const onSubmit = async (data: ScheduleFormData) => {
    if (selectedSources.length === 0) {
      setError("Please select at least one source");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const payload = {
        ...data,
        source_ids: selectedSources,
      };

      if (initialData?.id) {
        await api.put(`/api/schedules/${initialData.id}`, payload);
      } else {
        await api.post("/api/schedules", payload);
      }
      onSuccess();
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to save schedule");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      {error && (
        <div className="rounded-md bg-red-50 p-4">
          <p className="text-sm font-medium text-red-800">{error}</p>
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-gray-700">
          Schedule Name
        </label>
        <input
          {...register("name")}
          type="text"
          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
          placeholder="e.g., Daily News Digest"
        />
        {errors.name && (
          <p className="mt-1 text-sm text-red-600">{errors.name.message}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">
          When to run
        </label>
        <div className="mt-2 space-y-2">
          {CRON_PRESETS.map((preset) => (
            <label key={preset.value} className="flex items-center">
              <input
                type="radio"
                {...register("cron_expression")}
                value={preset.value}
                className="h-4 w-4 border-gray-300 text-blue-600"
              />
              <span className="ml-2 text-sm text-gray-700">{preset.label}</span>
            </label>
          ))}
        </div>

        <div className="mt-4 pt-4 border-t">
          <label className="flex items-center">
            <input
              type="radio"
              {...register("cron_expression")}
              value="custom"
              className="h-4 w-4 border-gray-300 text-blue-600"
            />
            <span className="ml-2 text-sm text-gray-700">Custom Cron Expression</span>
          </label>
          {cronValue === "custom" && (
            <input
              type="text"
              placeholder="0 6 * * *"
              className="mt-2 block w-full rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
            />
          )}
        </div>

        {errors.cron_expression && (
          <p className="mt-1 text-sm text-red-600">{errors.cron_expression.message}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">
          Timezone
        </label>
        <input
          {...register("timezone")}
          type="text"
          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
          placeholder="UTC"
        />
        {errors.timezone && (
          <p className="mt-1 text-sm text-red-600">{errors.timezone.message}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">
          Max Articles per Digest
        </label>
        <input
          {...register("max_articles", { valueAsNumber: true })}
          type="number"
          min="1"
          max="50"
          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
        />
        {errors.max_articles && (
          <p className="mt-1 text-sm text-red-600">{errors.max_articles.message}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Select Sources
        </label>
        {sources.length === 0 ? (
          <p className="text-sm text-gray-600">
            No sources available. Please create sources first.
          </p>
        ) : (
          <div className="space-y-2">
            {sources.map((source) => (
              <label key={source.id} className="flex items-center">
                <input
                  type="checkbox"
                  checked={selectedSources.includes(source.id)}
                  onChange={() => handleSourceToggle(source.id)}
                  className="h-4 w-4 rounded border-gray-300 text-blue-600"
                />
                <span className="ml-2 text-sm text-gray-700">{source.name}</span>
              </label>
            ))}
          </div>
        )}
      </div>

      <div className="flex space-x-4">
        <button
          type="submit"
          disabled={isSubmitting}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50"
        >
          {isSubmitting ? "Saving..." : "Save Schedule"}
        </button>
      </div>
    </form>
  );
}
