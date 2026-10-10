"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Camera,
  CheckCircle,
  Edit,
  Loader2,
  Plus,
  RefreshCw,
  Save,
  Shield,
  Trash2,
  X,
} from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { apiFetch, wsUrl } from "@/lib/api";

type Point = [number, number];
type Polygon = Point[];
interface DetectionRules {
  confidence: number;
  loitering_seconds: number;
  cooldown_seconds: number;
  inertia_frames: number;
  loitering_enabled: boolean;
  intrusion_enabled: boolean;
  concealment_enabled: boolean;
  bending_enabled: boolean;
}
interface Zone {
  id: string;
  name: string;
  kind: "restricted" | "loitering";
  points: Polygon;
  loitering_seconds: number;
}
interface CameraData {
  id: string;
  name: string;
  source: string;
  status: "active" | "connecting" | "reconnecting" | "error" | "stopped";
  last_frame_at: string | null;
  reconnect_attempts: number;
  fps: number;
  error: string | null;
  roi_points: Polygon;
  detection: Partial<DetectionRules>;
  zones: Zone[];
  privacy_masks: Polygon[];
}
interface CameraFeed {
  camera_id: string;
  data: string;
}
type Notice = { text: string; type: "success" | "error" };
type DrawingTarget = { kind: "roi" } | { kind: "zone" | "mask"; index: number };
const DEFAULT_RULES: DetectionRules = {
  confidence: 0.35,
  loitering_seconds: 5,
  cooldown_seconds: 3,
  inertia_frames: 3,
  loitering_enabled: true,
  intrusion_enabled: true,
  concealment_enabled: true,
  bending_enabled: true,
};
const inputClass =
  "w-full bg-black/40 border border-glass-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand disabled:opacity-60";
const buttonClass =
  "inline-flex items-center justify-center gap-2 rounded-lg border border-glass-border px-3 py-2 text-sm hover:bg-white/5 disabled:opacity-50 disabled:cursor-not-allowed";

async function responseError(response: Response, fallback: string) {
  const data = await response.json().catch(() => null);
  return typeof data?.detail === "string" ? data.detail : fallback;
}
function errorText(error: unknown) {
  return error instanceof Error
    ? error.message
    : "Connection failed. Please try again.";
}
function NoticeBanner({ notice }: { notice: Notice }) {
  return (
    <div
      role={notice.type === "error" ? "alert" : "status"}
      className={`rounded-lg border p-3 flex gap-2 text-sm ${notice.type === "error" ? "text-danger border-danger/30 bg-danger/10" : "text-green-400 border-green-500/30 bg-green-500/10"}`}
    >
      {notice.type === "error" ? (
        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
      ) : (
        <CheckCircle className="w-4 h-4 shrink-0 mt-0.5" />
      )}
      {notice.text}
    </div>
  );
}

export default function CamerasPage() {
  const { canAdmin } = useAuth();
  const [cameras, setCameras] = useState<CameraData[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [name, setName] = useState("");
  const [source, setSource] = useState("");
  const [selectedCam, setSelectedCam] = useState<CameraData | null>(null);

  const fetchCameras = useCallback((signal?: AbortSignal) => {
    return apiFetch("/cameras", { signal })
      .then(async (response) => {
        if (!response.ok)
          throw new Error(
            await responseError(response, "Could not load camera status."),
          );
        return response.json() as Promise<CameraData[]>;
      })
      .then((data) => {
        if (!signal?.aborted) {
          setCameras(data);
          setLoadError("");
        }
      })
      .catch((error) => {
        if (!signal?.aborted) setLoadError(errorText(error));
      })
      .finally(() => {
        if (!signal?.aborted) setLoading(false);
      });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void fetchCameras(controller.signal);
    const interval = setInterval(
      () => void fetchCameras(controller.signal),
      10_000,
    );
    return () => {
      controller.abort();
      clearInterval(interval);
    };
  }, [fetchCameras]);

  async function addCamera(event: React.FormEvent) {
    event.preventDefault();
    if (!canAdmin || !name.trim() || !source.trim()) return;
    setBusy("add");
    setNotice(null);
    try {
      const response = await apiFetch("/cameras", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), source: source.trim() }),
      });
      if (!response.ok)
        throw new Error(await responseError(response, "Could not add camera."));
      setName("");
      setSource("");
      setNotice({
        type: "success",
        text: "Camera saved. Its connection status is shown below; offline cameras are retained for automatic reconnection.",
      });
      await fetchCameras();
    } catch (error) {
      setNotice({ type: "error", text: errorText(error) });
    } finally {
      setBusy(null);
    }
  }

  async function cameraAction(
    camera: CameraData,
    action: "reconnect" | "delete",
  ) {
    if (!canAdmin) return;
    if (
      action === "delete" &&
      !confirm(`Remove camera “${camera.name}” and its rules?`)
    )
      return;
    setBusy(camera.id);
    setNotice(null);
    try {
      const response = await apiFetch(
        `/cameras/${encodeURIComponent(camera.id)}${action === "reconnect" ? "/reconnect" : ""}`,
        { method: action === "reconnect" ? "POST" : "DELETE" },
      );
      if (!response.ok)
        throw new Error(
          await responseError(response, `Could not ${action} camera.`),
        );
      setNotice({
        type: "success",
        text:
          action === "reconnect"
            ? "Reconnection requested. Status refreshes every 10 seconds."
            : "Camera removed.",
      });
      await fetchCameras();
    } catch (error) {
      setNotice({ type: "error", text: errorText(error) });
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="max-w-6xl mx-auto pb-10 space-y-6">
      <header>
        <h2 className="text-3xl font-bold tracking-tight mb-2">
          Cameras & Detection Rules
        </h2>
        <p className="text-foreground/60">
          Monitor connection health, define zones, and tune each camera. Health
          refreshes every 10 seconds.
        </p>
      </header>
      {notice && <NoticeBanner notice={notice} />}
      {loadError && (
        <NoticeBanner
          notice={{
            type: "error",
            text: `${loadError} Displayed camera status may be outdated.`,
          }}
        />
      )}
      <div className={`grid gap-6 ${canAdmin ? "lg:grid-cols-3" : ""}`}>
        {canAdmin && (
          <section className="glass-panel p-6 h-fit">
            <h3 className="text-xl font-semibold flex items-center gap-2 mb-5">
              <Camera className="w-5 h-5 text-brand" />
              Add Camera
            </h3>
            <form onSubmit={addCamera} className="space-y-4">
              <label className="block text-sm space-y-2">
                <span>Camera name</span>
                <input
                  className={inputClass}
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Checkout desk"
                  maxLength={100}
                  required
                />
              </label>
              <label className="block text-sm space-y-2">
                <span>Camera source</span>
                <input
                  className={inputClass}
                  value={source}
                  onChange={(event) => setSource(event.target.value)}
                  placeholder="0 or rtsp://camera/stream"
                  autoComplete="off"
                  required
                />
              </label>
              <p className="text-xs text-foreground/50">
                Use a USB camera index or network stream URL. Credentials are
                hidden in the camera list.
              </p>
              <button
                className={`${buttonClass} w-full bg-brand text-white`}
                disabled={busy !== null}
                type="submit"
              >
                {busy === "add" ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Plus className="w-4 h-4" />
                )}
                Add Camera
              </button>
            </form>
          </section>
        )}
        <section
          className={`glass-panel p-6 ${canAdmin ? "lg:col-span-2" : ""}`}
        >
          <div className="flex items-center justify-between gap-3 mb-5">
            <h3 className="text-xl font-semibold">
              Configured Cameras{" "}
              <span className="text-sm text-foreground/50">
                ({cameras.length})
              </span>
            </h3>
            <button
              className={buttonClass}
              onClick={() => void fetchCameras()}
              aria-label="Refresh camera health"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
          {loading ? (
            <div
              className="py-10 flex justify-center"
              role="status"
              aria-label="Loading cameras"
            >
              <Loader2 className="w-7 h-7 animate-spin text-brand" />
            </div>
          ) : cameras.length === 0 ? (
            <p className="p-8 text-center text-foreground/50 border border-dashed border-glass-border rounded-lg">
              No cameras configured.
              {canAdmin && " Add a camera to start monitoring."}
            </p>
          ) : (
            <div className="space-y-4">
              {cameras.map((camera) => (
                <article
                  className="rounded-lg border border-glass-border p-4 bg-black/20"
                  key={camera.id}
                >
                  <div className="flex flex-wrap justify-between gap-2">
                    <h4 className="font-semibold break-all">{camera.name}</h4>
                    <span
                      className={`text-xs rounded-full px-2.5 py-1 capitalize ${camera.status === "active" ? "text-green-400 bg-green-500/10" : camera.status === "error" ? "text-danger bg-danger/10" : "text-amber-300 bg-amber-400/10"}`}
                    >
                      {camera.status}
                    </span>
                  </div>
                  <p className="text-xs text-foreground/50 font-mono break-all mt-2">
                    {camera.source}
                  </p>
                  <dl className="grid sm:grid-cols-3 gap-3 my-4 text-xs">
                    <div>
                      <dt className="text-foreground/50">Last frame</dt>
                      <dd className="mt-1">
                        {camera.last_frame_at
                          ? new Date(camera.last_frame_at).toLocaleString()
                          : "No frame received"}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-foreground/50">Capture rate</dt>
                      <dd className="mt-1">
                        {Number(camera.fps || 0).toFixed(1)} FPS
                      </dd>
                    </div>
                    <div>
                      <dt className="text-foreground/50">Reconnect attempts</dt>
                      <dd className="mt-1">{camera.reconnect_attempts || 0}</dd>
                    </div>
                  </dl>
                  {camera.error && (
                    <p className="text-xs text-danger mb-3 break-words">
                      {camera.error}
                    </p>
                  )}
                  <p className="text-xs text-foreground/60 mb-3">
                    {camera.roi_points?.length
                      ? "Legacy ROI configured"
                      : "No legacy ROI"}{" "}
                    · {camera.zones?.length || 0} zones ·{" "}
                    {camera.privacy_masks?.length || 0} privacy masks
                  </p>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      className={buttonClass}
                      onClick={() => setSelectedCam(camera)}
                    >
                      <Edit className="w-4 h-4" />
                      {canAdmin ? "Configure" : "View Rules"}
                    </button>
                    {canAdmin && (
                      <>
                        <button
                          className={buttonClass}
                          disabled={busy !== null}
                          onClick={() => void cameraAction(camera, "reconnect")}
                        >
                          <RefreshCw
                            className={`w-4 h-4 ${busy === camera.id ? "animate-spin" : ""}`}
                          />
                          Reconnect
                        </button>
                        <button
                          className={`${buttonClass} text-danger ml-auto`}
                          disabled={busy !== null}
                          onClick={() => void cameraAction(camera, "delete")}
                          aria-label={`Delete ${camera.name}`}
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
      {selectedCam && (
        <CameraEditor
          key={selectedCam.id}
          camera={selectedCam}
          canAdmin={canAdmin}
          onClose={() => setSelectedCam(null)}
          onSaved={() => {
            setSelectedCam(null);
            setNotice({
              type: "success",
              text: "Camera rules, zones and privacy masks saved.",
            });
            void fetchCameras();
          }}
        />
      )}
    </div>
  );
}

function CameraEditor({
  camera,
  canAdmin,
  onClose,
  onSaved,
}: {
  camera: CameraData;
  canAdmin: boolean;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [name, setName] = useState(camera.name);
  const [rules, setRules] = useState<DetectionRules>({
    ...DEFAULT_RULES,
    ...camera.detection,
  });
  const [roi, setRoi] = useState<Polygon>(camera.roi_points || []);
  const [zones, setZones] = useState<Zone[]>(camera.zones || []);
  const [masks, setMasks] = useState<Polygon[]>(camera.privacy_masks || []);
  const [target, setTarget] = useState<DrawingTarget>({ kind: "roi" });
  const [frame, setFrame] = useState<string | null>(null);
  const [streamState, setStreamState] = useState("Connecting to preview…");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [dirty, setDirty] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const points =
    target.kind === "roi"
      ? roi
      : target.kind === "zone"
        ? zones[target.index]?.points || []
        : masks[target.index] || [];

  useEffect(() => {
    let stopped = false;
    let socket: WebSocket | undefined;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let retries = 0;
    const connect = () => {
      if (stopped) return;
      const connection = new WebSocket(wsUrl("/ws"));
      socket = connection;
      connection.onopen = () => {
        if (stopped || socket !== connection) return;
        retries = 0;
        setStreamState("Waiting for this camera’s next frame…");
      };
      connection.onmessage = (event) => {
        if (stopped || socket !== connection) return;
        try {
          const payload = JSON.parse(event.data);
          if (payload.type !== "multi_frame" || !Array.isArray(payload.cameras))
            return;
          const match = payload.cameras.find(
            (item: CameraFeed) => item.camera_id === camera.id,
          );
          if (match?.data) {
            setFrame(match.data);
            setStreamState("Live preview");
          }
        } catch {
          setStreamState("Preview unavailable. Waiting for a valid frame…");
        }
      };
      connection.onclose = (event) => {
        if (stopped || socket !== connection) return;
        if ([1008, 4401, 4403].includes(event.code)) {
          stopped = true;
          setStreamState("Camera preview authorization expired. Sign in again.");
          window.dispatchEvent(new Event("theftguard:unauthorized"));
          return;
        }
        setStreamState("Preview disconnected. Reconnecting…");
        retry = setTimeout(
          connect,
          Math.min(15000, 1000 * 2 ** Math.min(retries++, 4)),
        );
      };
      connection.onerror = () => {
        if (!stopped) connection.close();
      };
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onclose = null;
        socket.onerror = null;
        socket.close();
      }
    };
  }, [camera.id]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    let cancelled = false;
    function polygon(
      shape: Polygon,
      color: string,
      label: string,
      selected: boolean,
      mask = false,
    ) {
      if (!context || shape.length === 0) return;
      context.beginPath();
      context.moveTo(shape[0][0], shape[0][1]);
      shape.slice(1).forEach(([x, y]) => context.lineTo(x, y));
      context.closePath();
      context.fillStyle = mask ? "rgba(0,0,0,0.85)" : `${color}33`;
      context.fill();
      context.strokeStyle = color;
      context.lineWidth = selected ? 5 : 2;
      context.stroke();
      context.font = "bold 18px sans-serif";
      context.fillStyle = color;
      context.textAlign = "left";
      context.fillText(label, shape[0][0] + 10, Math.max(22, shape[0][1] - 10));
      if (selected)
        shape.forEach(([x, y], index) => {
          context.beginPath();
          context.arc(x, y, 6, 0, Math.PI * 2);
          context.fillStyle = "white";
          context.fill();
          context.fillText(String(index + 1), x + 10, y + 20);
        });
    }
    const overlay = () => {
      polygon(roi, "#60a5fa", "Legacy ROI", target.kind === "roi");
      zones.forEach((zone, index) =>
        polygon(
          zone.points,
          zone.kind === "restricted" ? "#f87171" : "#fbbf24",
          zone.name,
          target.kind === "zone" && target.index === index,
        ),
      );
      masks.forEach((mask, index) =>
        polygon(
          mask,
          "#c084fc",
          `Privacy ${index + 1}`,
          target.kind === "mask" && target.index === index,
          true,
        ),
      );
    };
    context.fillStyle = "#151824";
    context.fillRect(0, 0, 1280, 720);
    if (frame) {
      const picture = new Image();
      picture.onload = () => {
        if (!cancelled) {
          context.drawImage(picture, 0, 0, 1280, 720);
          overlay();
        }
      };
      picture.src = `data:image/jpeg;base64,${frame}`;
    } else {
      context.fillStyle = "#94a3b8";
      context.font = "22px sans-serif";
      context.textAlign = "center";
      context.fillText("No camera preview available", 640, 340);
      context.font = "16px sans-serif";
      context.fillText("You can still configure rules and regions.", 640, 375);
      overlay();
    }
    return () => {
      cancelled = true;
    };
  }, [frame, roi, zones, masks, target]);

  function updatePoints(next: Polygon) {
    if (!canAdmin || saving) return;
    setDirty(true);
    if (target.kind === "roi") setRoi(next);
    else if (target.kind === "zone")
      setZones((previous) =>
        previous.map((zone, index) =>
          index === target.index ? { ...zone, points: next } : zone,
        ),
      );
    else
      setMasks((previous) =>
        previous.map((mask, index) => (index === target.index ? next : mask)),
      );
  }
  function changeRule<K extends keyof DetectionRules>(
    key: K,
    value: DetectionRules[K],
  ) {
    setRules((previous) => ({ ...previous, [key]: value }));
    setDirty(true);
  }
  function changeZone(update: Partial<Zone>) {
    if (target.kind !== "zone") return;
    setZones((previous) =>
      previous.map((zone, index) =>
        index === target.index ? { ...zone, ...update } : zone,
      ),
    );
    setDirty(true);
  }
  function addShape(kind: "zone" | "mask") {
    if (kind === "zone") {
      setTarget({ kind: "zone", index: zones.length });
      setZones((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          name: `Zone ${previous.length + 1}`,
          kind: "restricted",
          points: [],
          loitering_seconds: 5,
        },
      ]);
    } else {
      setTarget({ kind: "mask", index: masks.length });
      setMasks((previous) => [...previous, []]);
    }
    setDirty(true);
  }
  function close() {
    if (!saving && (!dirty || confirm("Discard unsaved camera changes?")))
      onClose();
  }
  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (!canAdmin) return;
    if (
      (roi.length > 0 && roi.length < 3) ||
      zones.some((zone) => zone.points.length < 3) ||
      masks.some((mask) => mask.length < 3)
    ) {
      setError(
        "Each region needs at least 3 points. Clear the legacy ROI to disable it, or remove incomplete zones and masks.",
      );
      return;
    }
    if (!name.trim() || zones.some((zone) => !zone.name.trim())) {
      setError("Camera and zone names cannot be empty.");
      return;
    }
    setSaving(true);
    setError("");
    try {
      const response = await apiFetch(
        `/cameras/${encodeURIComponent(camera.id)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: name.trim(),
            detection: rules,
            roi_points: roi,
            zones: zones.map((zone) => ({ ...zone, name: zone.name.trim() })),
            privacy_masks: masks,
          }),
        },
      );
      if (!response.ok)
        throw new Error(
          await responseError(response, "Could not save camera changes."),
        );
      onSaved();
    } catch (saveError) {
      setError(errorText(saveError));
      setSaving(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-black/80 backdrop-blur-md"
      role="dialog"
      aria-modal="true"
      aria-labelledby="camera-editor-title"
    >
      <form
        onSubmit={save}
        className="glass-panel w-full max-w-6xl max-h-[94vh] overflow-y-auto shadow-2xl"
      >
        <div className="glass-header p-5 flex items-center justify-between gap-3">
          <div>
            <h3 id="camera-editor-title" className="text-xl font-bold">
              {camera.name}: Rules & Regions
            </h3>
            <p className="text-xs text-foreground/60 mt-1">
              {streamState} · Coordinates use the 1280 × 720 camera frame.
            </p>
          </div>
          <button
            type="button"
            className={buttonClass}
            disabled={saving}
            onClick={close}
            aria-label="Close camera editor"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <fieldset
          disabled={!canAdmin || saving}
          className="p-5 space-y-5 min-w-0"
        >
          <label className="block text-sm space-y-2">
            <span>Camera name</span>
            <input
              className={inputClass}
              value={name}
              maxLength={100}
              required
              onChange={(event) => {
                setName(event.target.value);
                setDirty(true);
              }}
            />
          </label>
          <section className="grid lg:grid-cols-[1fr_270px] gap-5">
            <div className="space-y-3">
              <p className="text-sm text-foreground/70">
                Select a region, then click the preview to place its corners.
                Regions need at least 3 points. Privacy masks obscure video,
                event images and clips.
              </p>
              <canvas
                ref={canvasRef}
                width={1280}
                height={720}
                aria-label="Camera region drawing canvas"
                className={`w-full aspect-video rounded-lg border border-glass-border ${canAdmin ? "cursor-crosshair" : ""}`}
                onClick={(event) => {
                  if (!canAdmin || saving) return;
                  const bounds = event.currentTarget.getBoundingClientRect();
                  updatePoints([
                    ...points,
                    [
                      Math.max(
                        0,
                        Math.min(
                          1280,
                          Math.round(
                            ((event.clientX - bounds.left) * 1280) /
                              bounds.width,
                          ),
                        ),
                      ),
                      Math.max(
                        0,
                        Math.min(
                          720,
                          Math.round(
                            ((event.clientY - bounds.top) * 720) /
                              bounds.height,
                          ),
                        ),
                      ),
                    ],
                  ]);
                }}
              />
              <div className="flex items-center flex-wrap gap-2">
                <span className="text-xs text-foreground/60 mr-auto">
                  {target.kind === "roi"
                    ? "Legacy ROI"
                    : target.kind === "zone"
                      ? zones[target.index]?.name
                      : `Privacy mask ${target.index + 1}`}{" "}
                  · {points.length} points
                </span>
                <button
                  type="button"
                  className={buttonClass}
                  disabled={!points.length || saving || !canAdmin}
                  onClick={() => updatePoints(points.slice(0, -1))}
                >
                  Undo Point
                </button>
                <button
                  type="button"
                  className={buttonClass}
                  onClick={() => updatePoints([])}
                >
                  Clear Points
                </button>
              </div>
              {target.kind === "zone" && zones[target.index] && (
                <div className="grid sm:grid-cols-3 gap-3 rounded-lg border border-glass-border p-3">
                  <label className="text-sm space-y-2">
                    <span>Zone name</span>
                    <input
                      className={inputClass}
                      value={zones[target.index].name}
                      maxLength={80}
                      required
                      onChange={(event) =>
                        changeZone({ name: event.target.value })
                      }
                    />
                  </label>
                  <label className="text-sm space-y-2">
                    <span>Zone rule</span>
                    <select
                      className={inputClass}
                      value={zones[target.index].kind}
                      onChange={(event) =>
                        changeZone({ kind: event.target.value as Zone["kind"] })
                      }
                    >
                      <option value="restricted">Restricted area</option>
                      <option value="loitering">Loitering</option>
                    </select>
                  </label>
                  {zones[target.index].kind === "loitering" && (
                    <label className="text-sm space-y-2">
                      <span>Dwell time (seconds)</span>
                      <input
                        type="number"
                        className={inputClass}
                        min={1}
                        max={3600}
                        step={1}
                        required
                        value={zones[target.index].loitering_seconds}
                        onChange={(event) =>
                          changeZone({
                            loitering_seconds: Number(event.target.value),
                          })
                        }
                      />
                    </label>
                  )}
                </div>
              )}
            </div>
            <div className="space-y-3">
              <h4 className="font-semibold">Regions</h4>
              <button
                type="button"
                className={`${buttonClass} w-full justify-start ${target.kind === "roi" ? "bg-brand/20 border-brand" : ""}`}
                onClick={() => setTarget({ kind: "roi" })}
              >
                Legacy ROI{" "}
                <span className="ml-auto text-xs">{roi.length} pts</span>
              </button>
              {zones.map((zone, index) => (
                <div className="flex gap-1" key={zone.id}>
                  <button
                    type="button"
                    className={`${buttonClass} flex-1 min-w-0 justify-start ${target.kind === "zone" && target.index === index ? "bg-brand/20 border-brand" : ""}`}
                    onClick={() => setTarget({ kind: "zone", index })}
                  >
                    <span className="truncate">
                      {zone.name || "Unnamed zone"}
                    </span>
                  </button>
                  <button
                    type="button"
                    className={`${buttonClass} text-danger`}
                    aria-label={`Delete ${zone.name}`}
                    onClick={() => {
                      setZones((previous) =>
                        previous.filter((_, item) => item !== index),
                      );
                      setTarget({ kind: "roi" });
                      setDirty(true);
                    }}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
              <button
                type="button"
                className={`${buttonClass} w-full`}
                onClick={() => addShape("zone")}
              >
                <Plus className="w-4 h-4" />
                Add Named Zone
              </button>
              <h4 className="font-semibold pt-2 flex gap-2 items-center">
                <Shield className="w-4 h-4" />
                Privacy Masks
              </h4>
              {masks.map((mask, index) => (
                <div className="flex gap-1" key={index}>
                  <button
                    type="button"
                    className={`${buttonClass} flex-1 justify-start ${target.kind === "mask" && target.index === index ? "bg-brand/20 border-brand" : ""}`}
                    onClick={() => setTarget({ kind: "mask", index })}
                  >
                    Mask {index + 1}{" "}
                    <span className="ml-auto text-xs">{mask.length} pts</span>
                  </button>
                  <button
                    type="button"
                    className={`${buttonClass} text-danger`}
                    aria-label={`Delete privacy mask ${index + 1}`}
                    onClick={() => {
                      setMasks((previous) =>
                        previous.filter((_, item) => item !== index),
                      );
                      setTarget({ kind: "roi" });
                      setDirty(true);
                    }}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
              <button
                type="button"
                className={`${buttonClass} w-full`}
                onClick={() => addShape("mask")}
              >
                <Plus className="w-4 h-4" />
                Add Privacy Mask
              </button>
              <p className="text-xs text-foreground/50">
                Blue: legacy ROI · Red: restricted · Amber: loitering · Purple:
                privacy. Changes apply after saving.
              </p>
            </div>
          </section>
          <section className="border-t border-glass-border pt-5 space-y-4">
            <h4 className="font-semibold">Detection Sensitivity</h4>
            <p className="text-sm text-foreground/60">
              Adjust persistence and cooldown to reduce repeated or short-lived
              alerts. Each enabled rule remains a review signal.
            </p>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <label className="text-sm space-y-2">
                <span>Confidence threshold</span>
                <input
                  className={inputClass}
                  type="number"
                  min={0.05}
                  max={0.95}
                  step={0.01}
                  required
                  value={rules.confidence}
                  onChange={(event) =>
                    changeRule("confidence", Number(event.target.value))
                  }
                />
              </label>
              <label className="text-sm space-y-2">
                <span>Default dwell time (s)</span>
                <input
                  className={inputClass}
                  type="number"
                  min={1}
                  max={3600}
                  required
                  value={rules.loitering_seconds}
                  onChange={(event) =>
                    changeRule("loitering_seconds", Number(event.target.value))
                  }
                />
              </label>
              <label className="text-sm space-y-2">
                <span>Alert cooldown (s)</span>
                <input
                  className={inputClass}
                  type="number"
                  min={0}
                  max={3600}
                  required
                  value={rules.cooldown_seconds}
                  onChange={(event) =>
                    changeRule("cooldown_seconds", Number(event.target.value))
                  }
                />
              </label>
              <label className="text-sm space-y-2">
                <span>Persistence (frames)</span>
                <input
                  className={inputClass}
                  type="number"
                  min={1}
                  max={120}
                  step={1}
                  required
                  value={rules.inertia_frames}
                  onChange={(event) =>
                    changeRule("inertia_frames", Number(event.target.value))
                  }
                />
              </label>
            </div>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {(
                [
                  ["intrusion_enabled", "Restricted-area alerts"],
                  ["loitering_enabled", "Loitering alerts"],
                  ["concealment_enabled", "Concealment alerts"],
                  ["bending_enabled", "Bending alerts"],
                ] as const
              ).map(([key, label]) => (
                <label key={key} className="flex gap-2 items-center text-sm">
                  <input
                    type="checkbox"
                    checked={rules[key]}
                    onChange={(event) => changeRule(key, event.target.checked)}
                  />
                  {label}
                </label>
              ))}
            </div>
          </section>
        </fieldset>
        <div className="p-5 border-t border-glass-border space-y-3">
          {error && <NoticeBanner notice={{ type: "error", text: error }} />}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-foreground/60" aria-live="polite">
              {!canAdmin
                ? "Read-only access. An administrator can change camera rules."
                : saving
                  ? "Saving changes…"
                  : dirty
                    ? "You have unsaved changes."
                    : "Camera configuration is up to date."}
            </p>
            <div className="flex gap-2">
              <button
                type="button"
                className={buttonClass}
                onClick={close}
                disabled={saving}
              >
                {canAdmin ? "Cancel" : "Close"}
              </button>
              {canAdmin && (
                <button
                  type="submit"
                  className={`${buttonClass} bg-brand text-white`}
                  disabled={saving || !dirty}
                >
                  {saving ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Save className="w-4 h-4" />
                  )}
                  Save Changes
                </button>
              )}
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}
