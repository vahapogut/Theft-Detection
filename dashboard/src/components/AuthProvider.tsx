"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { apiFetch } from "@/lib/api";

export type UserRole = "admin" | "operator" | "viewer";
export interface AuthUser {
  id: number;
  username: string;
  role: UserRole;
  active: boolean;
  created_at: string;
}

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;
  refresh: () => Promise<AuthUser | null>;
  logout: () => Promise<void>;
  canOperate: boolean;
  canAdmin: boolean;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth requires AuthProvider");
  return context;
}

export default function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const sessionRequest = useRef(0);
  const pathname = usePathname();
  const router = useRouter();

  const refresh = useCallback(() => {
    const request = ++sessionRequest.current;
    return apiFetch("/auth/me").then(async (response) => {
      if (request !== sessionRequest.current) return null;
      if (response.status === 401) {
        setUser(null);
        setError("");
        return null;
      }
      if (!response.ok) throw new Error("Oturum doğrulanamadı. Sunucu bağlantısını kontrol edin.");
      const account: AuthUser = await response.json();
      if (request !== sessionRequest.current) return null;
      setUser(account);
      setError("");
      return account;
    }).catch((cause: unknown) => {
      if (request === sessionRequest.current) setError(cause instanceof Error ? cause.message : "Sunucuya bağlanılamadı.");
      return null;
    }).finally(() => {
      if (request === sessionRequest.current) setLoading(false);
    });
  }, []);

  useEffect(() => {
    void refresh();
    const onUnauthorized = () => {
      sessionRequest.current += 1;
      setUser(null);
      setError("");
      setLoading(false);
    };
    const onFocus = () => { void refresh(); };
    window.addEventListener("theftguard:unauthorized", onUnauthorized);
    window.addEventListener("focus", onFocus);
    const interval = window.setInterval(() => { void refresh(); }, 60_000);
    return () => {
      window.removeEventListener("theftguard:unauthorized", onUnauthorized);
      window.removeEventListener("focus", onFocus);
      window.clearInterval(interval);
    };
  }, [refresh]);

  useEffect(() => {
    if (loading || error) return;
    if (!user && pathname !== "/login") router.replace("/login");
    if (user && pathname === "/login") router.replace("/");
  }, [loading, error, user, pathname, router]);

  const logout = useCallback(async () => {
    const response = await apiFetch("/auth/logout", { method: "POST" });
    if (!response.ok && response.status !== 401) throw new Error("Çıkış yapılamadı. Lütfen tekrar deneyin.");
    sessionRequest.current += 1;
    setUser(null);
    setError("");
    setLoading(false);
    router.replace("/login");
  }, [router]);

  const value: AuthContextValue = {
    user,
    loading,
    refresh,
    logout,
    canOperate: user?.role === "admin" || user?.role === "operator",
    canAdmin: user?.role === "admin",
  };

  let content = children;
  if (pathname !== "/login") {
    if (error) {
      content = <main className="m-auto max-w-md p-8 text-center" role="alert"><p>{error}</p><button type="button" onClick={() => { void refresh(); }} className="mt-4 rounded-lg bg-brand px-4 py-2 text-white">Tekrar dene</button></main>;
    } else if (loading || !user) {
      content = <main className="m-auto p-8 text-foreground/60" role="status">Oturum kontrol ediliyor…</main>;
    }
  }
  return <AuthContext.Provider value={value}>{content}</AuthContext.Provider>;
}
