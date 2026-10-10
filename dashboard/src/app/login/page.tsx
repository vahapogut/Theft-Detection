"use client";

import { useEffect, useState } from "react";
import { LockKeyhole } from "lucide-react";
import { apiFetch } from "@/lib/api";
import { useAuth } from "@/components/AuthProvider";

export default function LoginPage() {
  const { refresh } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [initialized, setInitialized] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    apiFetch("/auth/status", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("Sunucuya bağlanılamadı.");
        const status: { initialized: boolean } = await response.json();
        setInitialized(status.initialized);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Sunucuya bağlanılamadı.");
      });
    return () => controller.abort();
  }, []);

  async function login(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const response = await apiFetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      setPassword("");
      if (!response.ok) {
        if (response.status === 429) throw new Error("Çok fazla giriş denemesi. Bir süre sonra tekrar deneyin.");
        if (response.status === 401) throw new Error("Kullanıcı adı veya parola hatalı.");
        throw new Error("Giriş yapılamadı. Sunucu bağlantısını kontrol edin.");
      }
      const account = await refresh();
      if (!account) throw new Error("Oturum açılamadı. Tarayıcınızın çerez ayarlarını kontrol edin.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Sunucuya bağlanılamadı.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-[85vh] items-center justify-center">
      <section className="glass-panel w-full max-w-md p-8">
        <LockKeyhole className="mb-5 h-10 w-10 text-brand" aria-hidden="true" />
        <h1 className="text-3xl font-bold"><span className="text-brand">Theft</span>Guard</h1>
        <p className="mt-2 mb-7 text-sm text-foreground/60">Güvenlik paneline giriş yapın.</p>
        {initialized === false ? (
          <div role="status" className="rounded-lg border border-brand/30 bg-brand/10 p-4 text-sm leading-relaxed">
            Henüz yönetici hesabı oluşturulmadı. İlk hesabı oluşturması için sunucu yöneticinizle iletişime geçin.
          </div>
        ) : (
          <form onSubmit={login} className="space-y-5">
            <label className="block text-sm">Kullanıcı adı
              <input autoComplete="username" required minLength={3} maxLength={64} value={username} onChange={(event) => setUsername(event.target.value)} className="mt-2 w-full rounded-lg border border-glass-border bg-black/20 px-3 py-3 outline-none focus:border-brand" />
            </label>
            <label className="block text-sm">Parola
              <input type="password" autoComplete="current-password" required maxLength={1024} value={password} onChange={(event) => setPassword(event.target.value)} className="mt-2 w-full rounded-lg border border-glass-border bg-black/20 px-3 py-3 outline-none focus:border-brand" />
            </label>
            <button disabled={busy || initialized === null} type="submit" className="w-full rounded-lg bg-brand px-4 py-3 font-medium text-white disabled:opacity-50">{busy ? "Giriş yapılıyor…" : "Giriş yap"}</button>
          </form>
        )}
        {error && <p role="alert" className="mt-4 text-sm text-red-400">{error}</p>}
      </section>
    </div>
  );
}
