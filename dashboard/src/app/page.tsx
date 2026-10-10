"use client";

import { useEffect, useState } from "react";
import CameraGrid from "@/components/CameraGrid";
import StatsCharts from "@/components/StatsCharts";
import { apiFetch } from "@/lib/api";
import { Activity, Camera, ShieldAlert, Users } from "lucide-react";

interface CameraSummary { id: string; status: string }
interface FaceSummary { id: string; name: string; type: "blacklist" | "whitelist" }
interface MetricsSummary { weekly_data: number[]; cpu_load: number | null }
interface Overview { active: number; total: number; today: number; registeredFaces: number; cpu: number | null }

export default function Home() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function loadOverview() {
      try {
        const responses = await Promise.all([
          apiFetch("/cameras", { signal: controller.signal }),
          apiFetch("/stats", { signal: controller.signal }),
          apiFetch("/faces", { signal: controller.signal }),
        ]);
        const failed = responses.find((response) => !response.ok);
        if (failed) throw new Error(`Dashboard data could not be loaded (${failed.status}).`);
        const [cameras, metrics, faces]: [CameraSummary[], MetricsSummary, FaceSummary[]] = await Promise.all([
          responses[0].json(), responses[1].json(), responses[2].json(),
        ]);
        if (!controller.signal.aborted) {
          setOverview({
            active: cameras.filter((camera) => camera.status === "active").length,
            total: cameras.length,
            today: metrics.weekly_data[metrics.weekly_data.length - 1] ?? 0,
            registeredFaces: faces.length,
            cpu: typeof metrics.cpu_load === "number" && Number.isFinite(metrics.cpu_load) ? metrics.cpu_load : null,
          });
          setError("");
        }
      } catch (cause) {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Dashboard data could not be loaded.");
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(loadOverview, 10000);
      }
    }
    void loadOverview();
    return () => { controller.abort(); clearTimeout(timer); };
  }, []);

  const monitoring = Boolean(overview?.active) && !error;
  return (
    <div className="mx-auto max-w-7xl pb-10">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="mb-2 text-3xl font-bold tracking-tight">Dashboard Overview</h2>
          <p className="text-foreground/60">Live surveillance feeds and recorded system analytics.</p>
        </div>
        <div className={`flex items-center gap-3 text-sm font-medium ${monitoring ? "text-green-400" : "text-foreground/60"}`}>
          <span className={`h-3 w-3 rounded-full ${monitoring ? "bg-green-400" : "bg-foreground/40"}`} />
          {error ? "Status unavailable" : !overview ? "Loading status…" : monitoring ? "Monitoring active" : "No connected cameras"}
        </div>
      </header>
      {error && <p role="status" className="mb-5 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-300">{error} {overview ? "Last successful measurements are shown. " : ""}Retrying in 10 seconds.</p>}
      <div className="mb-8 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "Active Cameras", value: overview ? `${overview.active}/${overview.total}` : "—", icon: Camera, color: "text-blue-400" },
          { label: "Today's Alerts (UTC)", value: overview?.today.toLocaleString() ?? "—", icon: ShieldAlert, color: "text-danger" },
          { label: "Registered Faces", value: overview?.registeredFaces.toLocaleString() ?? "—", icon: Users, color: "text-purple-400" },
          { label: "CPU Usage", value: overview?.cpu === null ? "Unavailable" : overview ? `${overview.cpu.toFixed(1)}%` : "—", icon: Activity, color: "text-green-400" },
        ].map((stat) => {
          const Icon = stat.icon;
          return <div key={stat.label} className="glass-panel flex items-center gap-4 p-4"><div className={`rounded-lg bg-black/20 p-3 ${stat.color}`}><Icon className="h-6 w-6" /></div><div><div className="text-2xl font-bold">{stat.value}</div><div className="text-sm text-foreground/60">{stat.label}</div></div></div>;
        })}
      </div>
      <h3 className="mb-4 text-xl font-semibold">Live Feeds</h3>
      <CameraGrid />
      <div className="mt-8"><StatsCharts /></div>
    </div>
  );
}
