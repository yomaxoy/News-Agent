"use client";

import { useSession } from "next-auth/react";
import Link from "next/link";

export default function DashboardPage() {
  const { data: session } = useSession();

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Welcome back, {session?.user?.email?.split("@")[0]}!
        </h1>
        <p className="mt-2 text-gray-600">
          Manage your news sources and digest schedules
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <QuickActionCard
          title="Add Source"
          description="Add new RSS feeds to your collection"
          href="/dashboard/sources/new"
          icon="➕"
          color="blue"
        />
        <QuickActionCard
          title="Create Schedule"
          description="Set up a new digest generation schedule"
          href="/dashboard/schedules/new"
          icon="🕐"
          color="green"
        />
        <QuickActionCard
          title="View Digests"
          description="Browse previously generated digests"
          href="/dashboard/digests"
          icon="📚"
          color="purple"
        />
      </div>

      <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-2">
        <StatCard title="Active Sources" value="0" />
        <StatCard title="Schedules" value="0" />
      </div>
    </div>
  );
}

function QuickActionCard({
  title,
  description,
  href,
  icon,
  color,
}: {
  title: string;
  description: string;
  href: string;
  icon: string;
  color: string;
}) {
  const colorClasses = {
    blue: "bg-blue-50 text-blue-600 hover:bg-blue-100",
    green: "bg-green-50 text-green-600 hover:bg-green-100",
    purple: "bg-purple-50 text-purple-600 hover:bg-purple-100",
  };

  return (
    <Link
      href={href}
      className={`block p-6 rounded-lg border border-gray-200 shadow-sm transition-colors ${
        colorClasses[color as keyof typeof colorClasses]
      }`}
    >
      <div className="flex items-start space-x-4">
        <span className="text-3xl">{icon}</span>
        <div>
          <h3 className="font-semibold">{title}</h3>
          <p className="text-sm opacity-75">{description}</p>
        </div>
      </div>
    </Link>
  );
}

function StatCard({ title, value }: { title: string; value: string }) {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <p className="text-gray-600 text-sm">{title}</p>
      <p className="text-3xl font-bold text-gray-900 mt-2">{value}</p>
    </div>
  );
}
