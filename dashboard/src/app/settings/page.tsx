"use client";

import { useEffect, useState } from "react";
import {
  AlertCircle,
  Bell,
  CheckCircle2,
  Database,
  Film,
  Loader2,
  Mail,
  Save,
  Send,
  Shield,
  Trash2,
} from "lucide-react";
import { useAuth } from "@/components/AuthProvider";
import { apiFetch } from "@/lib/api";

interface Settings {
  emailEnabled: boolean;
  smtpServer: string;
  smtpPort: string;
  senderEmail: string;
  senderPassword: string;
  receiverEmail: string;
  telegramEnabled: boolean;
  telegramBotToken: string;
  telegramChatId: string;
  roiPoints: number[][];
  showHeatmap: boolean;
  includeImagesInNotifications: boolean;
  clipEnabled: boolean;
  clipPreSeconds: number;
  clipPostSeconds: number;
  retentionDays: number;
}
interface StorageInfo {
  storage: { event_count: number; file_count: number; bytes: number };
  recording: {
    available: boolean;
    ffmpeg_available: boolean;
    h264_available: boolean;
    enabled: boolean;
    pre_seconds: number;
    post_seconds: number;
    fps: number;
    max_buffer_bytes: number;
    max_cameras: number;
    max_pending: number;
    active_recordings: number;
    buffered_bytes: number;
  };
}
interface RetentionPreview {
  days: number;
  cutoff: string;
  token: string;
  event_count: number;
  file_count: number;
  bytes: number;
  protected_events: number;
}
type Notice = { text: string; type: "success" | "error" };
const DEFAULT_SETTINGS: Settings = {
  emailEnabled: false,
  smtpServer: "smtp.gmail.com",
  smtpPort: "587",
  senderEmail: "",
  senderPassword: "",
  receiverEmail: "",
  telegramEnabled: false,
  telegramBotToken: "",
  telegramChatId: "",
  roiPoints: [],
  showHeatmap: false,
  includeImagesInNotifications: true,
  clipEnabled: true,
  clipPreSeconds: 5,
  clipPostSeconds: 5,
  retentionDays: 0,
};
const inputClass =
  "w-full bg-black/40 border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand disabled:opacity-50";
const buttonClass =
  "inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-glass-border text-sm font-medium hover:bg-white/5 disabled:opacity-50 disabled:cursor-not-allowed";

function bytesLabel(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  const unit = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), 3);
  return `${(bytes / 1024 ** unit).toFixed(1)} ${["B", "KB", "MB", "GB"][unit]}`;
}
async function checkedJson<T>(
  response: Response,
  fallback: string,
): Promise<T> {
  const data = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(typeof data?.detail === "string" ? data.detail : fallback);
  if (!data) throw new Error(fallback);
  return data as T;
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
      className={`p-3 rounded-lg flex items-start gap-2 text-sm border ${notice.type === "success" ? "text-green-400 bg-green-500/10 border-green-500/30" : "text-danger bg-danger/10 border-danger/30"}`}
    >
      {notice.type === "success" ? (
        <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
      ) : (
        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
      )}
      {notice.text}
    </div>
  );
}

export default function SettingsPage() {
  const { canAdmin, loading: authLoading } = useAuth();
  const [loading, setLoading] = useState(true);
  const [loadFailed, setLoadFailed] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [storage, setStorage] = useState<StorageInfo | null>(null);
  const [storageNotice, setStorageNotice] = useState<Notice | null>(null);
  const [preview, setPreview] = useState<RetentionPreview | null>(null);
  const [cleanupBusy, setCleanupBusy] = useState<"preview" | "apply" | null>(
    null,
  );
  const [reload, setReload] = useState(0);

  useEffect(() => {
    if (!canAdmin) return;
    const controller = new AbortController();
    async function load() {
      const results = await Promise.allSettled([
        apiFetch("/settings", { signal: controller.signal }).then((response) =>
          checkedJson<Partial<Settings>>(response, "Could not load settings."),
        ),
        apiFetch("/storage", { signal: controller.signal }).then((response) =>
          checkedJson<StorageInfo>(response, "Could not load storage details."),
        ),
      ]);
      if (controller.signal.aborted) return;
      if (results[0].status === "fulfilled") {
        setSettings({ ...DEFAULT_SETTINGS, ...results[0].value });
        setLoadFailed(false);
      } else {
        setNotice({ type: "error", text: errorText(results[0].reason) });
        setLoadFailed(true);
      }
      if (results[1].status === "fulfilled") {
        setStorage(results[1].value);
        setStorageNotice(null);
      } else
        setStorageNotice({ type: "error", text: errorText(results[1].reason) });
      setLoading(false);
    }
    void load();
    return () => controller.abort();
  }, [canAdmin, reload]);

  function change<K extends keyof Settings>(key: K, value: Settings[K]) {
    setSettings((previous) => ({ ...previous, [key]: value }));
    setDirty(true);
    if (key === "retentionDays") {
      setPreview(null);
      setStorageNotice(null);
    }
  }
  async function refreshStorage() {
    try {
      setStorage(
        await checkedJson<StorageInfo>(
          await apiFetch("/storage"),
          "Could not refresh storage details.",
        ),
      );
    } catch (error) {
      setStorageNotice({ type: "error", text: errorText(error) });
    }
  }
  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (!canAdmin || loadFailed) return;
    setSaving(true);
    setNotice(null);
    try {
      await checkedJson(
        await apiFetch("/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(settings),
        }),
        "Could not save settings.",
      );
      setSettings((previous) => ({
        ...previous,
        senderPassword: previous.senderPassword ? "********" : "",
        telegramBotToken: previous.telegramBotToken ? "********" : "",
      }));
      setDirty(false);
      setNotice({
        type: "success",
        text: "Settings saved. Storage cleanup only runs when explicitly confirmed below.",
      });
      await refreshStorage();
    } catch (error) {
      setNotice({ type: "error", text: errorText(error) });
    } finally {
      setSaving(false);
    }
  }
  async function testNotifications() {
    if (!canAdmin) return;
    setTesting(true);
    setNotice(null);
    try {
      const data = await checkedJson<{
        status?: string;
        message?: string;
        results?: string[];
      }>(
        await apiFetch("/settings/test", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(settings),
        }),
        "Could not send test notification.",
      );
      const failed = data.status === "error";
      setNotice({
        type: failed ? "error" : "success",
        text:
          data.message ||
          data.results?.join(" ") ||
          "Test notification request completed. Check the enabled delivery channels.",
      });
    } catch (error) {
      setNotice({ type: "error", text: errorText(error) });
    } finally {
      setTesting(false);
    }
  }
  async function previewCleanup() {
    if (!canAdmin || settings.retentionDays <= 0) return;
    setCleanupBusy("preview");
    setStorageNotice(null);
    setPreview(null);
    try {
      const data = await checkedJson<RetentionPreview>(
        await apiFetch(
          `/storage/retention/preview?days=${settings.retentionDays}`,
        ),
        "Could not preview retention cleanup.",
      );
      if (!data.token || !data.cutoff)
        throw new Error("Cleanup preview is incomplete. Please try again.");
      setPreview(data);
    } catch (error) {
      setStorageNotice({ type: "error", text: errorText(error) });
    } finally {
      setCleanupBusy(null);
    }
  }
  async function applyCleanup() {
    if (!canAdmin || !preview || preview.days !== settings.retentionDays)
      return;
    if (
      !confirm(
        `Permanently delete ${preview.event_count} non-archived events and ${preview.file_count} media files older than ${new Date(preview.cutoff).toLocaleString()}? This cannot be undone.`,
      )
    )
      return;
    setCleanupBusy("apply");
    setStorageNotice(null);
    try {
      const data = await checkedJson<{
        deleted_events: number;
        deleted_files: number;
        errors?: string[];
      }>(
        await apiFetch("/storage/retention/apply", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            days: preview.days,
            cutoff: preview.cutoff,
            token: preview.token,
          }),
        }),
        "Could not apply retention cleanup. Preview again before retrying.",
      );
      setStorageNotice({
        type: data.errors?.length ? "error" : "success",
        text: `Deleted ${data.deleted_events} events and ${data.deleted_files} media files.${data.errors?.length ? ` ${data.errors.length} files could not be removed. Refresh the preview to review remaining files.` : " Archived events were preserved."}`,
      });
      await refreshStorage();
    } catch (error) {
      setStorageNotice({ type: "error", text: errorText(error) });
    } finally {
      setPreview(null);
      setCleanupBusy(null);
    }
  }

  if (authLoading || (canAdmin && loading))
    return (
      <div
        className="flex justify-center items-center h-[60vh]"
        role="status"
        aria-label="Loading settings"
      >
        <Loader2 className="w-8 h-8 animate-spin text-brand" />
      </div>
    );
  if (!canAdmin)
    return (
      <div className="max-w-4xl mx-auto glass-panel p-8">
        <Shield className="w-8 h-8 text-brand mb-4" />
        <h2 className="text-2xl font-bold mb-2">
          Administrator Access Required
        </h2>
        <p className="text-foreground/60">
          Notification credentials, recording settings and storage cleanup are
          managed by administrators.
        </p>
      </div>
    );
  return (
    <div className="max-w-4xl mx-auto pb-10 space-y-6">
      <header>
        <h2 className="text-3xl font-bold tracking-tight mb-2">
          Notifications, Recording & Storage
        </h2>
        <p className="text-foreground/60">
          Control alert delivery, event clips and how long you keep recorded
          evidence.
        </p>
      </header>
      {notice && <NoticeBanner notice={notice} />}
      {loadFailed ? (
        <button
          className={buttonClass}
          onClick={() => {
            setLoading(true);
            setNotice(null);
            setReload((value) => value + 1);
          }}
        >
          Retry Loading Settings
        </button>
      ) : (
        <form onSubmit={save} className="space-y-6">
          <fieldset
            disabled={saving || testing || cleanupBusy !== null}
            className="space-y-6 min-w-0"
          >
            <section className="glass-panel p-6 space-y-5">
              <h3 className="text-xl font-semibold flex gap-3 items-center">
                <Send className="w-5 h-5 text-blue-400" />
                Telegram Alerts
              </h3>
              <p className="text-sm text-foreground/60">
                Send alerts to your configured chat. Saved tokens are hidden;
                leave the token blank or unchanged to keep it.
              </p>
              <div className="grid sm:grid-cols-2 gap-4">
                <label className="text-sm space-y-2">
                  <span>Bot token</span>
                  <input
                    className={inputClass}
                    type="password"
                    autoComplete="new-password"
                    value={settings.telegramBotToken}
                    onChange={(event) =>
                      change("telegramBotToken", event.target.value)
                    }
                  />
                </label>
                <label className="text-sm space-y-2">
                  <span>Chat ID</span>
                  <input
                    className={inputClass}
                    value={settings.telegramChatId}
                    onChange={(event) =>
                      change("telegramChatId", event.target.value)
                    }
                  />
                </label>
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={settings.telegramEnabled}
                  onChange={(event) =>
                    change("telegramEnabled", event.target.checked)
                  }
                />
                Enable Telegram alerts
              </label>
            </section>
            <section className="glass-panel p-6 space-y-5">
              <h3 className="text-xl font-semibold flex gap-3 items-center">
                <Mail className="w-5 h-5 text-purple-400" />
                Email Alerts
              </h3>
              <div className="grid sm:grid-cols-2 gap-4">
                <label className="text-sm space-y-2">
                  <span>SMTP server</span>
                  <input
                    className={inputClass}
                    value={settings.smtpServer}
                    onChange={(event) =>
                      change("smtpServer", event.target.value)
                    }
                  />
                </label>
                <label className="text-sm space-y-2">
                  <span>SMTP port</span>
                  <input
                    className={inputClass}
                    type="number"
                    min={1}
                    max={65535}
                    value={settings.smtpPort}
                    onChange={(event) => change("smtpPort", event.target.value)}
                  />
                </label>
                <label className="text-sm space-y-2">
                  <span>Sender email</span>
                  <input
                    className={inputClass}
                    type="email"
                    value={settings.senderEmail}
                    onChange={(event) =>
                      change("senderEmail", event.target.value)
                    }
                  />
                </label>
                <label className="text-sm space-y-2">
                  <span>Password / app password</span>
                  <input
                    className={inputClass}
                    type="password"
                    autoComplete="new-password"
                    value={settings.senderPassword}
                    onChange={(event) =>
                      change("senderPassword", event.target.value)
                    }
                  />
                </label>
                <label className="text-sm space-y-2 sm:col-span-2">
                  <span>Recipient email(s)</span>
                  <input
                    className={inputClass}
                    value={settings.receiverEmail}
                    onChange={(event) =>
                      change("receiverEmail", event.target.value)
                    }
                  />
                </label>
              </div>
              <p className="text-xs text-foreground/50">
                Leave the password blank or unchanged to preserve the saved
                credential.
              </p>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={settings.emailEnabled}
                  onChange={(event) =>
                    change("emailEnabled", event.target.checked)
                  }
                />
                Enable email alerts
              </label>
            </section>
            <section className="glass-panel p-6 space-y-4">
              <h3 className="text-xl font-semibold flex gap-3 items-center">
                <Bell className="w-5 h-5 text-brand" />
                Notification Privacy
              </h3>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={settings.includeImagesInNotifications}
                  onChange={(event) =>
                    change("includeImagesInNotifications", event.target.checked)
                  }
                />
                Include event images in notifications
              </label>
              <p className="text-sm text-foreground/60">
                Turn this off to send text-only alerts. Camera privacy masks
                also apply to images and recorded clips.
              </p>
              <button
                type="button"
                className={buttonClass}
                disabled={!settings.emailEnabled && !settings.telegramEnabled}
                onClick={() => void testNotifications()}
              >
                {testing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                Send Test Notification
              </button>
              <p className="text-xs text-foreground/50">
                This button sends a real test message through enabled channels
                using the current form values.
              </p>
            </section>
            <section className="glass-panel p-6 space-y-4">
              <h3 className="text-xl font-semibold flex gap-3 items-center">
                <Film className="w-5 h-5 text-brand" />
                Event Clips
              </h3>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={settings.clipEnabled}
                  onChange={(event) =>
                    change("clipEnabled", event.target.checked)
                  }
                />
                Record a clip around each event
              </label>
              <div className="grid sm:grid-cols-2 gap-4">
                <label className="text-sm space-y-2">
                  <span>Seconds before event (0–15)</span>
                  <input
                    className={inputClass}
                    type="number"
                    min={0}
                    max={15}
                    step={1}
                    required
                    value={settings.clipPreSeconds}
                    onChange={(event) =>
                      change("clipPreSeconds", Number(event.target.value))
                    }
                  />
                </label>
                <label className="text-sm space-y-2">
                  <span>Seconds after event (1–30)</span>
                  <input
                    className={inputClass}
                    type="number"
                    min={1}
                    max={30}
                    step={1}
                    required
                    value={settings.clipPostSeconds}
                    onChange={(event) =>
                      change("clipPostSeconds", Number(event.target.value))
                    }
                  />
                </label>
              </div>
              {storage && (
                <div
                  className={`text-sm rounded-lg p-3 ${storage.recording.available ? "bg-green-500/10 text-green-400" : "bg-amber-400/10 text-amber-300"}`}
                >
                  {storage.recording.available
                    ? `Clip recording is available. ${storage.recording.active_recordings} active recording(s).`
                    : "Clip recording is unavailable on this server. Event snapshots can still be saved."}
                  <p className="text-xs mt-1">
                    FFmpeg:{" "}
                    {storage.recording.ffmpeg_available
                      ? "available"
                      : "unavailable"}{" "}
                    · H.264:{" "}
                    {storage.recording.h264_available
                      ? "available"
                      : "unavailable"}
                  </p>
                </div>
              )}
              <p className="text-xs text-foreground/50">
                Clips appear in event history after recording completes.
                Pre-event footage is limited by how long the camera has been
                running and available buffer space.
              </p>
            </section>
            <section className="glass-panel p-6 space-y-4">
              <h3 className="text-xl font-semibold flex gap-3 items-center">
                <Database className="w-5 h-5 text-brand" />
                Storage Retention
              </h3>
              {storage && (
                <dl className="grid grid-cols-3 gap-3 text-sm">
                  <div>
                    <dt className="text-foreground/50">Events</dt>
                    <dd className="text-xl font-semibold mt-1">
                      {storage.storage.event_count}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-foreground/50">Media files</dt>
                    <dd className="text-xl font-semibold mt-1">
                      {storage.storage.file_count}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-foreground/50">Media size</dt>
                    <dd className="text-xl font-semibold mt-1">
                      {bytesLabel(storage.storage.bytes)}
                    </dd>
                  </div>
                </dl>
              )}
              <label className="block text-sm space-y-2 max-w-sm">
                <span>Keep events for (days, 0 disables cleanup)</span>
                <input
                  className={inputClass}
                  type="number"
                  min={0}
                  max={3650}
                  step={1}
                  required
                  value={settings.retentionDays}
                  onChange={(event) =>
                    change("retentionDays", Number(event.target.value))
                  }
                />
              </label>
              <p className="text-sm text-foreground/60">
                Cleanup runs only after you preview and confirm it. Archived
                events and their media are protected. Save settings to remember
                the retention period.
              </p>
              <button
                className={buttonClass}
                type="button"
                disabled={
                  !Number.isInteger(settings.retentionDays) ||
                  settings.retentionDays < 1 ||
                  settings.retentionDays > 3650
                }
                onClick={() => void previewCleanup()}
              >
                {cleanupBusy === "preview" ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Database className="w-4 h-4" />
                )}
                Preview Cleanup
              </button>
              {preview && (
                <div className="p-4 rounded-lg border border-amber-400/30 bg-amber-400/5 space-y-3">
                  <p className="text-sm">
                    This will remove{" "}
                    <strong>{preview.event_count} events</strong> and{" "}
                    <strong>{preview.file_count} media files</strong> (
                    {bytesLabel(preview.bytes)}) older than{" "}
                    <strong>{new Date(preview.cutoff).toLocaleString()}</strong>
                    .
                  </p>
                  <p className="text-xs text-foreground/60">
                    {preview.protected_events || 0} archived events are
                    protected. Deletion is permanent.
                  </p>
                  <button
                    type="button"
                    className={`${buttonClass} border-danger/40 text-danger`}
                    disabled={
                      preview.event_count === 0 && preview.file_count === 0
                    }
                    onClick={() => void applyCleanup()}
                  >
                    {cleanupBusy === "apply" ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Trash2 className="w-4 h-4" />
                    )}
                    Permanently Delete Previewed Events
                  </button>
                </div>
              )}
              {storageNotice && <NoticeBanner notice={storageNotice} />}
            </section>
          </fieldset>
          <div className="flex flex-wrap justify-between gap-3 items-center">
            <p className="text-sm text-foreground/60" aria-live="polite">
              {saving
                ? "Saving settings…"
                : dirty
                  ? "You have unsaved changes."
                  : "Settings are up to date."}
            </p>
            <button
              type="submit"
              className={`${buttonClass} bg-brand text-white`}
              disabled={saving || testing || cleanupBusy !== null || !dirty}
            >
              {saving ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Save className="w-4 h-4" />
              )}
              Save Changes
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
