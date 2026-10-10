"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from "recharts";
import { Cpu, HardDrive, Loader2 } from "lucide-react";
import { apiFetch } from "@/lib/api";

interface SecurityStats {
  dates?: string[];
  weekly_data: number[];
  weekly_reviewed: number[];
  weekly_false_alarms: number[];
  hourly_data: number[];
  totals: { new: number; reviewed: number; false_alarm: number; total: number };
  cpu_load: number | null;
  ram_load: number | null;
}

interface WeeklyPoint {
  name: string;
  total: number;
  reviewed: number;
  falseAlarms: number;
}

interface HourlyPoint {
  time: string;
  alerts: number;
}

const tooltipStyle = { backgroundColor: "rgba(15, 17, 26, 0.97)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", color: "#f8fafc" };

function resourceValue(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;
}

export default function StatsCharts() {
  const [stats, setStats] = useState<SecurityStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function loadStats() {
      try {
        const response = await apiFetch("/stats", { signal: controller.signal });
        if (!response.ok) throw new Error(`İstatistikler alınamadı (${response.status}).`);
        const data: SecurityStats = await response.json();
        if (!controller.signal.aborted) {
          setStats(data);
          setError(null);
        }
      } catch (cause) {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "İstatistikler alınamadı.");
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
          timer = setTimeout(loadStats, 10000);
        }
      }
    }
    void loadStats();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  const weeklyData: WeeklyPoint[] = stats ? stats.weekly_data.map((total, index) => {
    const day = stats.dates?.[index] ? new Date(`${stats.dates[index]}T12:00:00Z`) : new Date();
    if (!stats.dates?.[index]) day.setUTCDate(day.getUTCDate() - (stats.weekly_data.length - index - 1));
    return {
      name: day.toLocaleDateString("tr-TR", { day: "numeric", month: "short", timeZone: "UTC" }),
      total,
      reviewed: stats.weekly_reviewed[index] ?? 0,
      falseAlarms: stats.weekly_false_alarms[index] ?? 0,
    };
  }) : [];
  const hourlyData: HourlyPoint[] = stats ? Array.from({ length: 24 }, (_, hour) => ({
    time: `${hour.toString().padStart(2, "0")}:00`,
    alerts: stats.hourly_data[hour] ?? 0,
  })) : [];
  const resources = [
    { label: "CPU kullanımı", value: resourceValue(stats?.cpu_load), Icon: Cpu, color: "text-blue-400", background: "bg-blue-500/10", bar: "bg-blue-500" },
    { label: "Bellek (RAM) kullanımı", value: resourceValue(stats?.ram_load), Icon: HardDrive, color: "text-purple-400", background: "bg-purple-500/10", bar: "bg-purple-500" },
  ];

  return (
    <div className="space-y-6" aria-busy={loading}>
      {error && <p role="status" className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-sm text-amber-300">{error} {stats ? "Son başarılı ölçümler gösteriliyor. " : ""}10 saniye içinde yeniden denenecek.</p>}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {resources.map(({ label, value, Icon, color, background, bar }) => (
          <div key={label} className="glass-panel flex items-center justify-between gap-4 p-4">
            <div className="flex items-center gap-3">
              <div className={`rounded-lg p-3 ${background} ${color}`}><Icon className="h-5 w-5" /></div>
              <div><h4 className="text-sm font-medium text-foreground/75">{label}</h4><p className="text-2xl font-bold">{value === null ? (loading ? "…" : "Ölçüm yok") : `${value.toFixed(1)}%`}</p></div>
            </div>
            {value !== null && <div role="meter" aria-label={label} aria-valuenow={value} aria-valuemin={0} aria-valuemax={100} className="h-2 w-32 overflow-hidden rounded-full bg-black/40"><div className={`h-full transition-all duration-500 ${bar}`} style={{ width: `${value}%` }} /></div>}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {[
          { label: "Toplam olay", value: stats?.totals.total, color: "text-foreground" },
          { label: "İnceleme bekliyor", value: stats?.totals.new, color: "text-amber-400" },
          { label: "İncelendi", value: stats?.totals.reviewed, color: "text-green-400" },
          { label: "Yanlış alarm", value: stats?.totals.false_alarm, color: "text-blue-400" },
        ].map(({ label, value, color }) => <div key={label} className="glass-panel p-4"><p className="text-sm text-foreground/65">{label}</p><p className={`mt-1 text-2xl font-bold ${color}`}>{value === undefined ? "—" : value.toLocaleString("tr-TR")}</p></div>)}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="glass-panel p-5">
          <h3 className="mb-1 text-lg font-medium">Haftalık güvenlik olayları</h3>
          <p className="mb-4 text-xs text-foreground/60">İnceleme sonuçları kayıtlı olaylardan hesaplanır. Günler UTC saatine göredir.</p>
          <div className="flex h-64 items-center justify-center">
            {loading ? <Loader2 aria-label="İstatistikler yükleniyor" className="h-8 w-8 animate-spin text-brand" />
              : !stats ? <p className="text-sm text-foreground/60">İstatistikler yüklenemedi.</p>
                : !weeklyData.some((point) => point.total > 0) ? <p className="text-sm text-foreground/60">Son yedi günde kayıtlı olay yok.</p>
                  : <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={weeklyData} accessibilityLayer>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                      <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis allowDecimals={false} stroke="rgba(255,255,255,0.5)" fontSize={12} tickLine={false} axisLine={false} width={35} />
                      <Tooltip cursor={{ fill: "rgba(255,255,255,0.05)" }} contentStyle={tooltipStyle} />
                      <Legend wrapperStyle={{ fontSize: "12px" }} />
                      <Bar dataKey="total" fill="#f59e0b" radius={[4, 4, 0, 0]} name="Tüm olaylar" />
                      <Bar dataKey="reviewed" fill="#22c55e" radius={[4, 4, 0, 0]} name="İncelendi" />
                      <Bar dataKey="falseAlarms" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Yanlış alarm" />
                    </BarChart>
                  </ResponsiveContainer>}
          </div>
        </div>

        <div className="glass-panel p-5">
          <h3 className="mb-1 text-lg font-medium">Bugünün saatlik olayları</h3>
          <p className="mb-4 text-xs text-foreground/60">00:00–23:59 UTC arasındaki kayıtlı olay sayıları.</p>
          <div className="flex h-64 items-center justify-center">
            {loading ? <Loader2 aria-label="Saatlik istatistikler yükleniyor" className="h-8 w-8 animate-spin text-brand" />
              : !stats ? <p className="text-sm text-foreground/60">İstatistikler yüklenemedi.</p>
                : !hourlyData.some((point) => point.alerts > 0) ? <p className="text-sm text-foreground/60">Bugün henüz kayıtlı olay yok.</p>
                  : <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={hourlyData} accessibilityLayer>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                      <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" fontSize={12} tickLine={false} axisLine={false} interval={3} />
                      <YAxis allowDecimals={false} stroke="rgba(255,255,255,0.5)" fontSize={12} tickLine={false} axisLine={false} width={35} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Line type="linear" dataKey="alerts" name="Olaylar" stroke="#3b82f6" strokeWidth={3} dot={false} activeDot={{ r: 6, fill: "#3b82f6" }} />
                    </LineChart>
                  </ResponsiveContainer>}
          </div>
        </div>
      </div>
    </div>
  );
}
