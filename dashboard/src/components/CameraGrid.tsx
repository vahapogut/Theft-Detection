"use client";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { Camera, Maximize2, AlertTriangle, WifiOff, Volume2, VolumeX } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { wsUrl } from "@/lib/api";

interface CameraFeed {
  camera_id: string;
  name: string;
  data: string | null;
  status: "active" | "connecting" | "reconnecting" | "error" | "stopped";
}
interface AlertData { id: string; message: string; timestamp: string; camera_id: string }
interface WsPayload { type: string; cameras: CameraFeed[]; alert: AlertData | null; alerts?: AlertData[] }

const statusLabels: Record<CameraFeed["status"], string> = {
  active: "Live", connecting: "Connecting", reconnecting: "Reconnecting", error: "Offline", stopped: "Stopped",
};

export default function CameraGrid() {
  const { user } = useAuth();
  const userId = user?.id;
  const [cameras, setCameras] = useState<CameraFeed[]>([]);
  const [alert, setAlert] = useState<AlertData | null>(null);
  const [alertCameraIds, setAlertCameraIds] = useState<string[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [message, setMessage] = useState("");
  const [soundEnabled, setSoundEnabled] = useState(false);
  const soundEnabledRef = useRef(false);
  const audioContextsRef = useRef(new Set<AudioContext>());

  useEffect(() => {
    if (userId === undefined) return;
    let disposed = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let alertTimer: ReturnType<typeof setTimeout> | undefined;
    let retries = 0;
    const seenAlerts = new Set<string>();
    const audioTimers = new Set<ReturnType<typeof setTimeout>>();
    const audioContexts = audioContextsRef.current;

    function playSiren() {
      if (!soundEnabledRef.current || !window.AudioContext || audioContexts.size > 0) return;
      let context: AudioContext | undefined;
      try {
        context = new AudioContext();
        const audioContext = context;
        audioContexts.add(audioContext);
        const oscillator = audioContext.createOscillator();
        const gain = audioContext.createGain();
        const now = audioContext.currentTime;
        oscillator.type = "sine";
        oscillator.frequency.setValueAtTime(580, now);
        oscillator.frequency.linearRampToValueAtTime(950, now + 0.35);
        oscillator.frequency.linearRampToValueAtTime(580, now + 0.7);
        oscillator.frequency.linearRampToValueAtTime(950, now + 1.05);
        oscillator.frequency.linearRampToValueAtTime(580, now + 1.4);
        gain.gain.setValueAtTime(0.2, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 1.4);
        oscillator.connect(gain);
        gain.connect(audioContext.destination);
        const cleanup = () => {
          clearTimeout(timeout);
          audioTimers.delete(timeout);
          audioContexts.delete(audioContext);
          oscillator.disconnect();
          gain.disconnect();
          if (audioContext.state !== "closed") void audioContext.close().catch(() => {});
        };
        const timeout = setTimeout(cleanup, 1800);
        audioTimers.add(timeout);
        oscillator.onended = cleanup;
        if (audioContext.state === "suspended") void audioContext.resume().catch(() => {});
        oscillator.start(now);
        oscillator.stop(now + 1.4);
      } catch {
        if (context && context.state !== "closed") {
          audioContexts.delete(context);
          void context.close().catch(() => {});
        }
        if (!disposed) setMessage("Your browser could not play the alert sound. Visual alerts remain active.");
      }
    }

    function connect() {
      if (disposed) return;
      const connection = new WebSocket(wsUrl("/ws"));
      socket = connection;
      connection.onopen = () => {
        if (disposed || socket !== connection) return;
        retries = 0;
        setIsConnected(true);
        setMessage("");
      };
      connection.onmessage = (event: MessageEvent<string>) => {
        if (disposed || socket !== connection) return;
        try {
          const payload: WsPayload = JSON.parse(event.data);
          if (payload.type !== "multi_frame" || !Array.isArray(payload.cameras)) return;
          setCameras(payload.cameras);
          const newAlerts: AlertData[] = [];
          for (const incoming of payload.alerts ?? (payload.alert ? [payload.alert] : [])) {
            if (!incoming.id || seenAlerts.has(incoming.id)) continue;
            seenAlerts.add(incoming.id);
            newAlerts.push(incoming);
            if (seenAlerts.size > 1000) {
              const oldest = seenAlerts.values().next().value;
              if (oldest !== undefined) seenAlerts.delete(oldest);
            }
          }
          if (newAlerts.length > 0) {
            setAlert(newAlerts.find((incoming) => incoming.id === payload.alert?.id) ?? newAlerts[newAlerts.length - 1]);
            setAlertCameraIds((current) => [...new Set([...current, ...newAlerts.map((incoming) => incoming.camera_id)])]);
            playSiren();
            clearTimeout(alertTimer);
            alertTimer = setTimeout(() => {
              if (!disposed) { setAlert(null); setAlertCameraIds([]); }
            }, 5000);
          }
        } catch {
          setMessage("A camera update could not be read. Waiting for the next frame.");
        }
      };
      connection.onclose = (event) => {
        if (disposed || socket !== connection) return;
        setIsConnected(false);
        if ([1008, 4401, 4403].includes(event.code)) {
          window.dispatchEvent(new Event("theftguard:unauthorized"));
          return;
        }
        reconnectTimer = setTimeout(connect, Math.min(15000, 1000 * 2 ** Math.min(retries++, 4)));
      };
      connection.onerror = () => { if (!disposed) connection.close(); };
    }

    connect();
    return () => {
      disposed = true;
      clearTimeout(reconnectTimer);
      clearTimeout(alertTimer);
      for (const timer of audioTimers) clearTimeout(timer);
      for (const context of audioContexts) {
        if (context.state !== "closed") void context.close().catch(() => {});
      }
      audioContexts.clear();
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onclose = null;
        socket.onerror = null;
        socket.close();
      }
    };
  }, [userId]);

  function toggleSound() {
    const enabled = !soundEnabled;
    soundEnabledRef.current = enabled;
    setSoundEnabled(enabled);
    if (!enabled) {
      for (const context of audioContextsRef.current) {
        if (context.state !== "closed") void context.close().catch(() => {});
      }
      audioContextsRef.current.clear();
    }
  }

  async function toggleFullscreen(event: React.MouseEvent<HTMLButtonElement>) {
    const container = event.currentTarget.closest<HTMLElement>("[data-camera-feed]");
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (container?.requestFullscreen) await container.requestFullscreen();
      else setMessage("Fullscreen is unavailable in this browser.");
    } catch {
      setMessage("Your browser could not open this camera in fullscreen.");
    }
  }

  if (!user) return null;
  return (
    <div className="mb-6 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 text-sm">
        <p className="text-foreground/60">{isConnected ? "Connected to surveillance server" : "Connecting to surveillance server…"}</p>
        <button type="button" onClick={toggleSound} aria-pressed={soundEnabled} className="flex items-center gap-2 rounded-lg border border-glass-border px-3 py-2 hover:bg-white/5 focus-visible:outline-2 focus-visible:outline-brand">
          {soundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}{soundEnabled ? "Alert sound on" : "Enable alert sound"}
        </button>
      </div>
      {!isConnected && <div role="status" className="flex items-center gap-2 rounded border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-300"><WifiOff className="h-5 w-5 shrink-0" /><span>Live connection unavailable. Reconnecting automatically; displayed frames may be outdated.</span></div>}
      {message && <p role="status" className="text-sm text-amber-300">{message}</p>}
      {alert && <p role="alert" className="flex items-center gap-2 rounded border border-danger/30 bg-danger/10 p-3 text-sm text-danger"><AlertTriangle className="h-4 w-4 shrink-0" />{alert.message}</p>}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {cameras.length === 0 && isConnected ? <div className="rounded-lg border border-dashed border-glass-border p-10 text-center text-foreground/50 lg:col-span-2">No cameras are configured. <Link href="/cameras" className="text-brand underline">Open camera management</Link>.</div>
          : cameras.map((camera) => {
            const isAlerting = alertCameraIds.includes(camera.camera_id);
            const live = isConnected && camera.status === "active";
            return (
              <div key={camera.camera_id} data-camera-feed className={`glass-panel group relative overflow-hidden bg-black transition-all duration-300 ${isAlerting ? "alert-pulse ring-2 ring-danger" : ""}`}>
                <div className="glass-header absolute left-0 right-0 top-0 z-10 flex items-center justify-between gap-3 p-2">
                  <div className="flex min-w-0 items-center gap-2"><Camera className={`h-4 w-4 shrink-0 ${isAlerting ? "text-danger" : "text-brand"}`} /><span className="truncate text-sm font-semibold">{camera.name}</span></div>
                  <div className="flex shrink-0 items-center gap-2"><span className={`text-xs ${live ? "text-green-400" : "text-amber-300"}`}>{isConnected ? statusLabels[camera.status] || "Unknown status" : "Disconnected"}</span><button type="button" onClick={toggleFullscreen} aria-label={`Toggle fullscreen: ${camera.name}`} className="rounded p-1 hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-brand"><Maximize2 className="h-4 w-4 text-white/70" /></button></div>
                </div>
                <div className="relative flex aspect-video items-center justify-center overflow-hidden bg-black/40">
                  {camera.data ? <Image src={`data:image/jpeg;base64,${camera.data}`} alt={`${camera.name} camera feed`} fill unoptimized sizes="(min-width: 1024px) 50vw, 100vw" className={`object-contain ${live ? "" : "opacity-50"}`} /> : <p className="px-4 text-center text-sm text-foreground/50">{camera.status === "active" ? "Waiting for the first frame…" : `${statusLabels[camera.status] || "Unavailable"} — no current frame`}</p>}
                  {isAlerting && <div className="pointer-events-none absolute inset-0 z-20 border-4 border-danger/50" />}
                </div>
              </div>
            );
          })}
      </div>
    </div>
  );
}
