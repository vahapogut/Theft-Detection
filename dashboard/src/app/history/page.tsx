"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import { Archive, Download, Film, Image as ImageIcon, Loader2, Search, X } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { apiFetch, apiUrl } from "@/lib/api";

type ReviewStatus = "new" | "reviewed" | "false_alarm";

interface SecurityEvent {
  id: string;
  message: string;
  timestamp: string;
  image_path: string;
  camera_id: string;
  camera_name: string;
  event_type: string;
  status: ReviewStatus;
  notes: string;
  updated_at: string | null;
  reviewed_by: string | null;
  archived: boolean;
  clip_path: string | null;
}

interface EventPage {
  items: SecurityEvent[];
  total: number;
  page: number;
  page_size: number;
}

interface CameraOption {
  id: string;
  name: string;
}

const EVENT_TYPES = [
  ["suspicious_behavior", "Şüpheli davranış"],
  ["blacklisted_face", "Kara listedeki yüz"],
  ["item_concealment", "Nesne gizleme"],
  ["restricted_area", "Yasak bölge"],
  ["loitering", "Uzun süre bekleme"],
  ["bending", "Eğilme"],
] as const;

const STATUS_LABELS: Record<ReviewStatus, string> = {
  new: "Yeni",
  reviewed: "İncelendi",
  false_alarm: "Yanlış alarm",
};

const inputClass = "w-full rounded-lg border border-glass-border bg-black/40 px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-brand";
const buttonClass = "rounded-lg border border-glass-border px-3 py-2 text-sm hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-brand";

function formatTime(timestamp: string | null) {
  if (!timestamp) return "—";
  if (/^\d{8}_\d{6}$/.test(timestamp)) {
    return `${timestamp.slice(6, 8)}.${timestamp.slice(4, 6)}.${timestamp.slice(0, 4)} ${timestamp.slice(9, 11)}:${timestamp.slice(11, 13)}:${timestamp.slice(13, 15)}`;
  }
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? timestamp : date.toLocaleString("tr-TR");
}

async function responseError(response: Response, fallback: string) {
  const payload: unknown = await response.json().catch(() => null);
  if (payload && typeof payload === "object" && "detail" in payload && typeof payload.detail === "string") {
    return payload.detail;
  }
  return `${fallback} (${response.status})`;
}

function EventReview({ event, onSaved, onClose }: {
  event: SecurityEvent;
  onSaved: () => void;
  onClose: () => void;
}) {
  const [status, setStatus] = useState(event.status);
  const [notes, setNotes] = useState(event.notes || "");
  const [archived, setArchived] = useState(event.archived);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const changed = status !== event.status || notes !== (event.notes || "") || archived !== event.archived;

  async function saveReview(submitEvent: React.FormEvent<HTMLFormElement>) {
    submitEvent.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const response = await apiFetch(`/events/${encodeURIComponent(event.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status, notes, archived }),
      });
      if (!response.ok) throw new Error(await responseError(response, "İnceleme kaydedilemedi"));
      onSaved();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "İnceleme kaydedilemedi.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <form onSubmit={saveReview} aria-label={`${event.id} olayını incele`} className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="font-semibold">Olay incelemesi</h3>
        <button type="button" onClick={onClose} disabled={saving} aria-label="İncelemeyi kapat" className={buttonClass}><X className="h-4 w-4" /></button>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="space-y-1 text-sm">
          <span>İnceleme durumu</span>
          <select value={status} onChange={(change) => setStatus(change.target.value as ReviewStatus)} disabled={saving} className={inputClass}>
            {Object.entries(STATUS_LABELS).map(([value, label]) => <option key={value} value={value} className="bg-[#0f111a]">{label}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-2 self-end py-2 text-sm">
          <input type="checkbox" checked={archived} onChange={(change) => setArchived(change.target.checked)} disabled={saving} className="h-4 w-4 accent-brand" />
          Arşivde koru
        </label>
      </div>
      <label className="block space-y-1 text-sm">
        <span>İnceleme notu</span>
        <textarea value={notes} onChange={(change) => setNotes(change.target.value)} disabled={saving} maxLength={10000} rows={3} className={inputClass} placeholder="Olayla ilgili gözlemlerinizi ekleyin." />
      </label>
      <p className="text-xs text-foreground/60">Arşivlenen olayların kanıtları otomatik saklama temizliğinden korunur.</p>
      {event.reviewed_by && <p className="text-xs text-foreground/60">Son inceleyen: {event.reviewed_by} · {formatTime(event.updated_at)}</p>}
      {error && <p role="alert" className="text-sm text-danger">{error}</p>}
      <div className="flex items-center gap-3">
        <button type="submit" disabled={!changed || saving} className={`${buttonClass} flex items-center gap-2 border-brand/30 bg-brand/20 text-brand`}>
          {saving && <Loader2 className="h-4 w-4 animate-spin" />}{saving ? "Kaydediliyor…" : "Değişiklikleri kaydet"}
        </button>
        <button type="button" onClick={onClose} disabled={saving} className={buttonClass}>Vazgeç</button>
      </div>
    </form>
  );
}

export default function HistoryPage() {
  const { canOperate } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");
  const [filters, setFilters] = useState({ search: "", camera_id: "", event_type: "", status: "", archived: "", start_date: "", end_date: "" });
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [revision, setRevision] = useState(0);
  const [result, setResult] = useState<{ key: string; data: EventPage | null; error: string | null } | null>(null);
  const [cameras, setCameras] = useState<CameraOption[]>([]);
  const [cameraError, setCameraError] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const invalidDates = Boolean(filters.start_date && filters.end_date && filters.start_date > filters.end_date);
  const searchPending = searchQuery.trim() !== filters.search;

  useEffect(() => {
    const timeout = setTimeout(() => {
      setFilters((current) => ({ ...current, search: searchQuery.trim() }));
      setPage(1);
      setExpandedId(null);
    }, 300);
    return () => clearTimeout(timeout);
  }, [searchQuery]);

  const filterQuery = useMemo(() => {
    const query = new URLSearchParams();
    for (const [name, value] of Object.entries(filters)) {
      if (value) query.set(name, value);
    }
    return query.toString();
  }, [filters]);
  const requestQuery = `${filterQuery}&page=${page}&page_size=${pageSize}`;
  const requestKey = `${requestQuery}&revision=${revision}`;
  const loading = !invalidDates && result?.key !== requestKey;
  const data = result?.key === requestKey ? result.data : null;
  const error = result?.key === requestKey ? result.error : null;

  useEffect(() => {
    if (invalidDates) return;
    const controller = new AbortController();
    async function loadEvents() {
      try {
        const response = await apiFetch(`/events?${requestQuery}`, { signal: controller.signal });
        if (!response.ok) throw new Error(await responseError(response, "Olaylar alınamadı"));
        const body: EventPage = await response.json();
        if (!controller.signal.aborted) {
          setResult({ key: requestKey, data: body, error: null });
          const lastPage = Math.max(1, Math.ceil(body.total / body.page_size));
          if (body.page > lastPage) setPage(lastPage);
        }
      } catch (cause) {
        if (!controller.signal.aborted) setResult({ key: requestKey, data: null, error: cause instanceof Error ? cause.message : "Olaylar alınamadı." });
      }
    }
    void loadEvents();
    return () => controller.abort();
  }, [requestQuery, requestKey, invalidDates]);

  useEffect(() => {
    const controller = new AbortController();
    async function loadCameras() {
      try {
        const response = await apiFetch("/cameras", { signal: controller.signal });
        if (!response.ok) throw new Error("Kamera listesi alınamadı.");
        const body: CameraOption[] = await response.json();
        if (!controller.signal.aborted) setCameras(body);
      } catch {
        if (!controller.signal.aborted) setCameraError(true);
      }
    }
    void loadCameras();
    return () => controller.abort();
  }, []);

  function updateFilter(name: keyof typeof filters, value: string) {
    setFilters((current) => ({ ...current, [name]: value }));
    setPage(1);
    setExpandedId(null);
    setNotice(null);
  }

  async function exportCSV() {
    setExporting(true);
    setExportError(null);
    try {
      const response = await apiFetch(`/events/export.csv?${filterQuery}`);
      if (!response.ok) throw new Error(await responseError(response, "CSV dışa aktarılamadı"));
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = `TheftGuard_Alarmlar_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) {
      setExportError(cause instanceof Error ? cause.message : "CSV dışa aktarılamadı.");
    } finally {
      setExporting(false);
    }
  }

  const total = data?.total ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const firstEntry = data?.items.length ? (page - 1) * pageSize + 1 : 0;
  const lastEntry = data?.items.length ? (page - 1) * pageSize + data.items.length : 0;

  return (
    <div className="mx-auto max-w-7xl pb-10">
      <header className="mb-8 flex flex-col justify-between gap-4 md:flex-row md:items-end">
        <div>
          <h2 className="mb-2 text-3xl font-bold tracking-tight">Olay geçmişi</h2>
          <p className="text-foreground/60">Güvenlik olaylarını filtreleyin, kanıtları inceleyin ve inceleme sonuçlarını kaydedin.</p>
        </div>
        <button onClick={exportCSV} disabled={exporting || loading || invalidDates || searchPending || !total} className={`${buttonClass} flex items-center justify-center gap-2 border-brand/35 bg-brand/20 font-semibold text-brand`}>
          {exporting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
          {exporting ? "Hazırlanıyor…" : "Filtrelenen tüm olayları CSV indir"}
        </button>
      </header>

      {!canOperate && <p className="mb-4 text-sm text-foreground/60">Gözlemci erişimi: Olayları görüntüleyebilir ve dışa aktarabilirsiniz. İnceleme değişiklikleri için operatör veya yönetici erişimi gerekir.</p>}
      {notice && <p role="status" className="mb-4 text-sm text-green-400">{notice}</p>}
      {exportError && <p role="alert" className="mb-4 text-sm text-danger">{exportError}</p>}

      <div className="glass-panel overflow-hidden">
        <div className="grid gap-4 border-b border-glass-border p-4 sm:grid-cols-2 lg:grid-cols-3">
          <label className="space-y-1 text-sm">
            <span>Ara</span>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-foreground/50" />
              <input type="search" value={searchQuery} onChange={(change) => setSearchQuery(change.target.value)} placeholder="Olay, kamera veya inceleme notu" className={`${inputClass} pl-9`} />
            </div>
          </label>
          <label className="space-y-1 text-sm">
            <span>Kamera</span>
            <select value={filters.camera_id} onChange={(change) => updateFilter("camera_id", change.target.value)} className={inputClass}>
              <option value="" className="bg-[#0f111a]">Tüm kameralar</option>
              {cameras.map((camera) => <option key={camera.id} value={camera.id} className="bg-[#0f111a]">{camera.name}</option>)}
            </select>
            {cameraError && <span role="status" className="text-xs text-amber-400">Kamera seçenekleri yüklenemedi.</span>}
          </label>
          <label className="space-y-1 text-sm">
            <span>Olay türü</span>
            <select value={filters.event_type} onChange={(change) => updateFilter("event_type", change.target.value)} className={inputClass}>
              <option value="" className="bg-[#0f111a]">Tüm olay türleri</option>
              {EVENT_TYPES.map(([value, label]) => <option key={value} value={value} className="bg-[#0f111a]">{label}</option>)}
            </select>
          </label>
          <label className="space-y-1 text-sm">
            <span>İnceleme durumu</span>
            <select value={filters.status} onChange={(change) => updateFilter("status", change.target.value)} className={inputClass}>
              <option value="" className="bg-[#0f111a]">Tüm durumlar</option>
              {Object.entries(STATUS_LABELS).map(([value, label]) => <option key={value} value={value} className="bg-[#0f111a]">{label}</option>)}
            </select>
          </label>
          <label className="space-y-1 text-sm">
            <span>Başlangıç tarihi</span>
            <input type="date" value={filters.start_date} max={filters.end_date || undefined} onChange={(change) => updateFilter("start_date", change.target.value)} className={inputClass} />
          </label>
          <label className="space-y-1 text-sm">
            <span>Bitiş tarihi</span>
            <input type="date" value={filters.end_date} min={filters.start_date || undefined} onChange={(change) => updateFilter("end_date", change.target.value)} className={inputClass} />
          </label>
          <label className="space-y-1 text-sm">
            <span>Arşiv</span>
            <select value={filters.archived} onChange={(change) => updateFilter("archived", change.target.value)} className={inputClass}>
              <option value="" className="bg-[#0f111a]">Tüm olaylar</option>
              <option value="true" className="bg-[#0f111a]">Arşivlenenler</option>
              <option value="false" className="bg-[#0f111a]">Arşivlenmeyenler</option>
            </select>
          </label>
          <div className="flex flex-wrap items-center gap-3 sm:col-span-2 lg:col-span-3">
            <button onClick={() => { setSearchQuery(""); setFilters({ search: "", camera_id: "", event_type: "", status: "", archived: "", start_date: "", end_date: "" }); setPage(1); setExpandedId(null); }} className={buttonClass}>Filtreleri temizle</button>
            <button onClick={() => { setRevision((current) => current + 1); setExpandedId(null); }} disabled={loading || invalidDates} className={buttonClass}>Yenile</button>
            <span className="text-xs text-foreground/60">Tarih aralığı UTC günlerine göre uygulanır.</span>
          </div>
        </div>

        {invalidDates && <p role="alert" className="p-4 text-sm text-danger">Başlangıç tarihi bitiş tarihinden sonra olamaz.</p>}
        {error && <div role="alert" className="flex items-center gap-3 p-4 text-sm text-danger"><span>{error}</span><button onClick={() => setRevision((current) => current + 1)} className={buttonClass}>Tekrar dene</button></div>}

        <div className="overflow-x-auto" aria-busy={loading}>
          <table className="w-full border-collapse text-left">
            <caption className="sr-only">Filtrelenmiş güvenlik olayları ve inceleme durumları</caption>
            <thead>
              <tr className="border-b border-glass-border bg-black/20 text-sm text-foreground/70">
                <th scope="col" className="p-4 font-medium">Tarih / Kamera</th>
                <th scope="col" className="p-4 font-medium">Olay</th>
                <th scope="col" className="p-4 font-medium">İnceleme</th>
                <th scope="col" className="p-4 font-medium">Kanıt</th>
                <th scope="col" className="p-4 font-medium">Ayrıntılar</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              {loading ? <tr><td colSpan={5} className="p-8 text-center text-foreground/60"><Loader2 className="mx-auto mb-2 h-6 w-6 animate-spin" /><span role="status">Olaylar yükleniyor…</span></td></tr>
                : !error && !invalidDates && !data?.items.length ? <tr><td colSpan={5} className="p-8 text-center text-foreground/60">Bu filtrelerle eşleşen olay bulunamadı.</td></tr>
                  : data?.items.map((event) => (
                    <Fragment key={event.id}>
                      <tr className="border-b border-glass-border/50 align-top hover:bg-white/[0.02]">
                        <td className="whitespace-nowrap p-4"><p>{formatTime(event.timestamp)}</p><p className="mt-1 text-foreground/60">{event.camera_name || event.camera_id || "Eski kayıt"}</p></td>
                        <td className="max-w-md p-4"><p className="font-medium">{EVENT_TYPES.find(([type]) => type === event.event_type)?.[1] || event.event_type}</p><p className="mt-1 break-words text-foreground/70">{event.message}</p><p title={event.id} className="mt-2 break-all font-mono text-xs text-foreground/45">{event.id}</p></td>
                        <td className="p-4"><span className={`inline-block whitespace-nowrap rounded px-2 py-1 text-xs font-semibold ${event.status === "false_alarm" ? "bg-blue-500/15 text-blue-400" : event.status === "reviewed" ? "bg-green-500/15 text-green-400" : "bg-amber-500/15 text-amber-400"}`}>{STATUS_LABELS[event.status]}</span>{event.archived && <span className="mt-2 flex items-center gap-1 text-xs text-foreground/60"><Archive className="h-3 w-3" />Arşivlendi</span>}</td>
                        <td className="p-4"><div className="flex flex-col gap-2">{event.image_path && <a href={apiUrl(`/${event.image_path.replace(/^\/+/, "")}`)} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 whitespace-nowrap text-brand hover:underline"><ImageIcon className="h-4 w-4" />Görüntü<span className="sr-only"> (yeni sekmede)</span></a>}{event.clip_path && <a href={apiUrl(`/${event.clip_path.replace(/^\/+/, "")}`)} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 whitespace-nowrap text-brand hover:underline"><Film className="h-4 w-4" />Video<span className="sr-only"> (yeni sekmede)</span></a>}{!event.image_path && !event.clip_path && <span className="text-foreground/50">Kanıt yok</span>}</div></td>
                        <td className="p-4"><button type="button" onClick={() => setExpandedId((current) => current === event.id ? null : event.id)} aria-expanded={expandedId === event.id} aria-controls={`review-${event.id}`} className={buttonClass}>{canOperate ? "İncele" : "Notları gör"}<span className="sr-only">: {event.id}</span></button></td>
                      </tr>
                      {expandedId === event.id && <tr id={`review-${event.id}`} className="border-b border-glass-border bg-black/20"><td colSpan={5} className="p-5">{canOperate ? <EventReview event={event} onClose={() => setExpandedId(null)} onSaved={() => { setExpandedId(null); setNotice("Olay incelemesi kaydedildi."); setRevision((current) => current + 1); }} /> : <div className="space-y-2"><p className="whitespace-pre-wrap break-words">{event.notes || "Bu olay için inceleme notu eklenmedi."}</p>{event.reviewed_by && <p className="text-xs text-foreground/60">Son inceleyen: {event.reviewed_by} · {formatTime(event.updated_at)}</p>}</div>}</td></tr>}
                    </Fragment>
                  ))}
            </tbody>
          </table>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 border-t border-glass-border p-4 text-sm text-foreground/60">
          <span aria-live="polite">{data ? `${total} olaydan ${firstEntry}–${lastEntry} arası` : "—"}</span>
          <label className="flex items-center gap-2">Sayfa başına<select value={pageSize} onChange={(change) => { setPageSize(Number(change.target.value)); setPage(1); setExpandedId(null); }} className={`${inputClass} w-auto`}>{[25, 50, 100].map((size) => <option value={size} key={size} className="bg-[#0f111a]">{size}</option>)}</select></label>
          <nav aria-label="Olay sayfaları" className="flex items-center gap-3">
            <button onClick={() => { setPage((current) => current - 1); setExpandedId(null); }} disabled={page <= 1 || loading || invalidDates} className={buttonClass}>Önceki</button>
            <span>Sayfa {page}{data ? ` / ${pageCount}` : ""}</span>
            <button onClick={() => { setPage((current) => current + 1); setExpandedId(null); }} disabled={!data || page >= pageCount || loading || invalidDates} className={buttonClass}>Sonraki</button>
          </nav>
        </div>
      </div>
    </div>
  );
}
