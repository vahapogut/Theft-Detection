"use client";

import { useCallback, useEffect, useState } from "react";
import { ShieldCheck, UserPlus } from "lucide-react";
import { apiFetch } from "@/lib/api";
import { AuthUser, UserRole, useAuth } from "@/components/AuthProvider";

const roleLabels: Record<UserRole, string> = { admin: "Yönetici", operator: "Operatör", viewer: "İzleyici" };
const inputClass = "w-full rounded-lg border border-glass-border bg-background px-3 py-2 outline-none focus:border-brand disabled:opacity-50";

async function requireSuccess(response: Response) {
  if (response.ok) return;
  const body: { detail?: unknown } = await response.json().catch(() => ({}));
  throw new Error(typeof body.detail === "string" ? body.detail : "İşlem tamamlanamadı. Lütfen tekrar deneyin.");
}

function RoleOptions() {
  return Object.entries(roleLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>);
}

function UserEditor({ account, lastAdmin, onSaved }: { account: AuthUser; lastAdmin: boolean; onSaved: (id: number) => Promise<void> }) {
  const [role, setRole] = useState<UserRole>(account.role);
  const [active, setActive] = useState(account.active);
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  async function save(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const response = await apiFetch(`/users/${account.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role, active, ...(password ? { password } : {}) }),
      });
      await requireSuccess(response);
      setPassword("");
      setMessage("Kullanıcı güncellendi. Değişiklik yapıldıysa açık oturumları sonlandırıldı.");
      await onSaved(account.id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "İşlem tamamlanamadı.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={save} className="glass-panel p-5">
      <div className="mb-4 flex items-center justify-between gap-3"><h3 className="font-semibold">{account.username}</h3><span className={`rounded-full px-2 py-1 text-xs ${account.active ? "bg-green-500/10 text-green-400" : "bg-gray-500/10 text-foreground/50"}`}>{account.active ? "Aktif" : "Devre dışı"}</span></div>
      <div className="grid gap-4 md:grid-cols-3">
        <label className="space-y-2 text-sm"><span>Rol</span><select aria-label={`${account.username} rolü`} value={role} disabled={busy || lastAdmin} onChange={(event) => setRole(event.target.value as UserRole)} className={inputClass}><RoleOptions /></select></label>
        <label className="space-y-2 text-sm"><span>Yeni parola (isteğe bağlı)</span><input aria-label={`${account.username} yeni parola`} type="password" autoComplete="new-password" minLength={12} maxLength={1024} value={password} disabled={busy} onChange={(event) => setPassword(event.target.value)} placeholder="En az 12 karakter" className={inputClass} /></label>
        <div className="flex items-end justify-between gap-3"><label className="flex items-center gap-2 pb-2 text-sm"><input type="checkbox" checked={active} disabled={busy || lastAdmin} onChange={(event) => setActive(event.target.checked)} />Aktif</label><button type="submit" disabled={busy} className="rounded-lg bg-brand/20 px-4 py-2 text-sm text-brand disabled:opacity-50">{busy ? "Kaydediliyor…" : "Kaydet"}</button></div>
      </div>
      {lastAdmin && <p className="mt-3 text-xs text-foreground/50">Son aktif yöneticinin rolü ve erişimi korunur.</p>}
      {error && <p role="alert" className="mt-3 text-sm text-red-400">{error}</p>}
      {message && <p role="status" className="mt-3 text-sm text-green-400">{message}</p>}
    </form>
  );
}

export default function UsersPage() {
  const { user, canAdmin, refresh } = useAuth();
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserRole>("viewer");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const loadUsers = useCallback((signal?: AbortSignal) => {
    return apiFetch("/users", { signal }).then(async (response) => {
      await requireSuccess(response);
      const body: { users: AuthUser[] } = await response.json();
      setUsers(body.users);
    }).catch((cause: unknown) => {
      if (!signal?.aborted) setError(cause instanceof Error ? cause.message : "Kullanıcılar yüklenemedi.");
    }).finally(() => {
      if (!signal?.aborted) setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!canAdmin) return;
    const controller = new AbortController();
    void loadUsers(controller.signal);
    return () => controller.abort();
  }, [canAdmin, loadUsers]);

  async function create(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const response = await apiFetch("/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password, role }),
      });
      await requireSuccess(response);
      setPassword("");
      setUsername("");
      setRole("viewer");
      setMessage("Kullanıcı oluşturuldu.");
      await loadUsers();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Kullanıcı oluşturulamadı.");
    } finally {
      setBusy(false);
    }
  }

  if (!canAdmin) return <p role="alert" className="p-6 text-foreground/60">Kullanıcı yönetimi için yönetici yetkisi gerekir.</p>;
  const activeAdmins = users.filter((account) => account.active && account.role === "admin").length;

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <header><h1 className="flex items-center gap-3 text-3xl font-bold"><ShieldCheck className="text-brand" />Kullanıcılar ve yetkiler</h1><p className="mt-2 text-foreground/60">Yönetici tüm ayarları yönetir. Operatör olayları değerlendirir. İzleyici görüntüleri ve olayları inceler.</p></header>
      <section className="glass-panel p-6">
        <h2 className="mb-5 flex items-center gap-2 text-lg font-semibold"><UserPlus className="h-5 w-5 text-brand" />Kullanıcı ekle</h2>
        <form onSubmit={create} className="grid items-end gap-4 md:grid-cols-4">
          <label className="space-y-2 text-sm"><span>Kullanıcı adı</span><input required minLength={3} maxLength={64} pattern="[a-zA-Z0-9_.\-]{3,64}" title="3–64 harf, rakam, nokta, tire veya alt çizgi" autoComplete="off" value={username} onChange={(event) => setUsername(event.target.value)} className={inputClass} /></label>
          <label className="space-y-2 text-sm"><span>Parola</span><input required type="password" minLength={12} maxLength={1024} autoComplete="new-password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="En az 12 karakter" className={inputClass} /></label>
          <label className="space-y-2 text-sm"><span>Rol</span><select value={role} onChange={(event) => setRole(event.target.value as UserRole)} className={inputClass}><RoleOptions /></select></label>
          <button disabled={busy} type="submit" className="rounded-lg bg-brand px-4 py-2 font-medium text-white disabled:opacity-50">{busy ? "Oluşturuluyor…" : "Kullanıcı oluştur"}</button>
        </form>
        {error && <p role="alert" className="mt-4 text-sm text-red-400">{error}</p>}
        {message && <p role="status" className="mt-4 text-sm text-green-400">{message}</p>}
      </section>
      <section className="space-y-4" aria-label="Kullanıcı hesapları">
        {loading ? <p role="status" className="text-foreground/60">Kullanıcılar yükleniyor…</p> : users.map((account) => <UserEditor key={`${account.id}-${account.role}-${account.active}`} account={account} lastAdmin={account.active && account.role === "admin" && activeAdmins === 1} onSaved={async (id) => {
          if (id === user?.id && !await refresh()) return;
          await loadUsers();
        }} />)}
      </section>
    </div>
  );
}
