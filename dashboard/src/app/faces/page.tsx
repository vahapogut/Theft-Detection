"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Users, UserPlus, Trash2, ShieldAlert, ShieldCheck, Loader2, AlertCircle, CheckCircle } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { apiFetch } from "@/lib/api";

interface FaceData { id: string; name: string; type: "blacklist" | "whitelist" }
type Notice = { text: string; type: "success" | "error" };
const inputClass = "w-full rounded-lg border border-glass-border bg-black/40 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand disabled:opacity-50";

async function responseError(response: Response, fallback: string) {
  const data: unknown = await response.json().catch(() => null);
  if (data && typeof data === "object") {
    if ("detail" in data && typeof data.detail === "string") return data.detail;
    if ("message" in data && typeof data.message === "string") return data.message;
  }
  return `${fallback} (${response.status})`;
}

export default function FacesPage() {
  const { canAdmin } = useAuth();
  const [faces, setFaces] = useState<FaceData[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [name, setName] = useState("");
  const [type, setType] = useState<FaceData["type"]>("blacklist");
  const [photo, setPhoto] = useState<File | null>(null);
  const [message, setMessage] = useState<Notice | null>(null);
  const [deleteLoadingId, setDeleteLoadingId] = useState<string | null>(null);
  const photoInputRef = useRef<HTMLInputElement>(null);

  const fetchFaces = useCallback((signal?: AbortSignal) => {
    return apiFetch("/faces", { signal }).then(async (response) => {
      if (!response.ok) throw new Error(await responseError(response, "Registered faces could not be loaded"));
      const data: FaceData[] = await response.json();
      if (!signal?.aborted) { setFaces(data); setLoadError(""); }
    }).catch((cause: unknown) => {
      if (!signal?.aborted) setLoadError(cause instanceof Error ? cause.message : "Registered faces could not be loaded.");
    }).finally(() => {
      if (!signal?.aborted) setLoading(false);
    });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void fetchFaces(controller.signal);
    return () => controller.abort();
  }, [fetchFaces]);

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (file && (!new Set(["image/png", "image/jpeg"]).has(file.type) || file.size > 10 * 1024 * 1024)) {
      setPhoto(null);
      event.target.value = "";
      setMessage({ type: "error", text: "Choose a PNG or JPEG image no larger than 10 MB." });
      return;
    }
    setPhoto(file);
    setMessage(null);
  }

  async function handleRegister(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canAdmin || submitting) return;
    if (!name.trim() || !photo) {
      setMessage({ text: "Enter a name and choose a reference photo.", type: "error" });
      return;
    }
    setSubmitting(true);
    setMessage(null);
    const formData = new FormData();
    formData.append("file", photo);
    formData.append("name", name.trim());
    formData.append("type", type);
    try {
      const response = await apiFetch("/faces/register", { method: "POST", body: formData });
      if (!response.ok) throw new Error(await responseError(response, "Face registration failed"));
      const result: { status: string; message?: string } = await response.json();
      if (result.status !== "success") throw new Error(result.message || "Face registration failed.");
      setMessage({ text: `Face registered: ${name.trim()}`, type: "success" });
      setName("");
      setPhoto(null);
      if (photoInputRef.current) photoInputRef.current.value = "";
      await fetchFaces();
    } catch (cause) {
      setMessage({ text: cause instanceof Error ? cause.message : "Face registration failed.", type: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(face: FaceData) {
    if (!canAdmin || deleteLoadingId || !confirm(`Delete the face registration for “${face.name}”?`)) return;
    setDeleteLoadingId(face.id);
    setMessage(null);
    try {
      const response = await apiFetch(`/faces/${encodeURIComponent(face.id)}`, { method: "DELETE" });
      if (!response.ok) throw new Error(await responseError(response, "Face registration could not be deleted"));
      setMessage({ text: "Face registration deleted.", type: "success" });
      await fetchFaces();
    } catch (cause) {
      setMessage({ text: cause instanceof Error ? cause.message : "Face registration could not be deleted.", type: "error" });
    } finally {
      setDeleteLoadingId(null);
    }
  }

  return (
    <div className="mx-auto max-w-6xl pb-10">
      <header className="mb-8">
        <h2 className="mb-2 text-3xl font-bold tracking-tight">Face Recognition Management</h2>
        <p className="text-foreground/60">Review the registered faces used for blacklist alerts and trusted-person matching.</p>
      </header>
      {!canAdmin && <p className="mb-5 text-sm text-foreground/60">Read-only access. Administrators can register and remove faces.</p>}
      {message && <div role={message.type === "error" ? "alert" : "status"} className={`mb-6 flex items-center gap-3 rounded-lg border p-4 ${message.type === "success" ? "border-green-500/30 bg-green-500/10 text-green-400" : "border-danger/30 bg-danger/10 text-danger"}`}>{message.type === "success" ? <CheckCircle className="h-5 w-5 shrink-0" /> : <AlertCircle className="h-5 w-5 shrink-0" />}<span className="text-sm font-medium">{message.text}</span></div>}
      {loadError && <div role="alert" className="mb-6 flex flex-wrap items-center gap-3 rounded-lg border border-danger/30 bg-danger/10 p-4 text-sm text-danger"><span>{loadError}</span><button type="button" onClick={() => { setLoading(true); void fetchFaces(); }} disabled={loading} className="rounded border border-danger/40 px-3 py-1 disabled:opacity-50">Retry</button></div>}
      <div className={`grid grid-cols-1 gap-6 ${canAdmin ? "lg:grid-cols-3" : ""}`}>
        {canAdmin && <section className="glass-panel h-fit p-6">
          <h3 className="mb-6 flex items-center gap-3 text-xl font-semibold"><UserPlus className="h-5 w-5 text-brand" />Register New Face</h3>
          <form onSubmit={handleRegister} className="space-y-4">
            <label className="block space-y-2 text-sm font-medium text-foreground/80"><span>Full Name</span><input type="text" value={name} onChange={(event) => setName(event.target.value)} placeholder="Person's name" maxLength={100} disabled={submitting} className={inputClass} required /></label>
            <label className="block space-y-2 text-sm font-medium text-foreground/80"><span>List Type</span><select value={type} onChange={(event) => setType(event.target.value as FaceData["type"])} disabled={submitting} className={inputClass}><option value="blacklist" className="bg-[#0f111a]">Blacklist (Alerts Enabled)</option><option value="whitelist" className="bg-[#0f111a]">Whitelist (Trusted)</option></select></label>
            <label className="block space-y-2 text-sm font-medium text-foreground/80"><span>Face Reference Photo</span><input ref={photoInputRef} type="file" accept="image/png,image/jpeg" onChange={handleFileChange} disabled={submitting} className={`${inputClass} file:mr-3 file:rounded file:border-0 file:bg-brand/20 file:px-2 file:py-1 file:text-brand`} required /><span className="block text-xs text-foreground/50">PNG or JPEG, up to 10 MB. Use a photo with one clear face.</span></label>
            {photo && <p className="break-all text-xs text-brand">Selected: {photo.name}</p>}
            <button type="submit" disabled={submitting || deleteLoadingId !== null} className="flex w-full items-center justify-center gap-2 rounded-lg bg-brand py-2 font-medium text-white hover:bg-brand/90 disabled:opacity-50">{submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <UserPlus className="h-4 w-4" />}{submitting ? "Registering…" : "Register Face"}</button>
          </form>
        </section>}
        <section className={`glass-panel p-6 ${canAdmin ? "lg:col-span-2" : ""}`} aria-busy={loading}>
          <h3 className="mb-6 flex items-center gap-3 text-xl font-semibold"><Users className="h-5 w-5 text-purple-400" />Registered Faces{!loading && !loadError ? ` (${faces.length})` : ""}</h3>
          {loading ? <div role="status" className="flex items-center justify-center gap-3 p-12 text-foreground/60"><Loader2 className="h-6 w-6 animate-spin text-brand" />Loading registered faces…</div>
            : faces.length === 0 ? <p className="rounded-lg border border-dashed border-glass-border p-12 text-center text-foreground/50">{loadError ? "The face list is unavailable." : "No faces registered yet."}</p>
              : <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {faces.map((face) => <div key={face.id} className={`glass-panel flex items-center justify-between gap-3 border-t-4 p-4 ${face.type === "blacklist" ? "border-t-danger/70" : "border-t-green-500/70"}`}>
                  <div className="flex min-w-0 items-center gap-3"><div className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full border-2 text-lg font-bold ${face.type === "blacklist" ? "border-danger/30 bg-danger/10 text-danger" : "border-green-500/30 bg-green-500/10 text-green-400"}`}>{face.name.charAt(0).toUpperCase()}</div><div className="min-w-0"><h4 className="break-words font-semibold">{face.name}</h4><span className={`mt-1 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${face.type === "blacklist" ? "bg-danger/10 text-danger" : "bg-green-500/10 text-green-400"}`}>{face.type === "blacklist" ? <ShieldAlert className="h-3 w-3" /> : <ShieldCheck className="h-3 w-3" />}{face.type === "blacklist" ? "Blacklist" : "Whitelist"}</span></div></div>
                  {canAdmin && <button type="button" onClick={() => { void handleDelete(face); }} disabled={deleteLoadingId !== null || submitting} aria-label={`Delete face registration: ${face.name}`} className="rounded-lg p-2 text-foreground/50 hover:bg-danger/10 hover:text-danger focus-visible:outline-2 focus-visible:outline-brand disabled:opacity-50">{deleteLoadingId === face.id ? <Loader2 className="h-5 w-5 animate-spin" /> : <Trash2 className="h-5 w-5" />}</button>}
                </div>)}
              </div>}
        </section>
      </div>
    </div>
  );
}
