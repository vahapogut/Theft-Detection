"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { LayoutDashboard, History, Settings, Users, Video, ShieldCheck, LogOut } from "lucide-react";
import { useAuth } from "@/components/AuthProvider";

export default function Sidebar() {
  const pathname = usePathname();
  const { user, logout, canAdmin, canOperate } = useAuth();
  const [error, setError] = useState("");
  const [loggingOut, setLoggingOut] = useState(false);

  if (pathname === "/login" || !user) return null;

  const links = [
    { href: "/", label: "Overview", icon: LayoutDashboard },
    { href: "/cameras", label: "Cameras", icon: Video },
    ...(canOperate ? [{ href: "/faces", label: "Faces", icon: Users }] : []),
    { href: "/history", label: "Alert History", icon: History },
    ...(canAdmin ? [{ href: "/settings", label: "Settings", icon: Settings }, { href: "/users", label: "Kullanıcılar", icon: ShieldCheck }] : []),
  ];

  return (
    <aside className="w-64 glass-panel border-l-0 rounded-l-none min-h-screen flex flex-col p-4">
      <div className="flex items-center mb-10 px-2 mt-4">
        <h1 className="text-xl font-bold tracking-wider">
          <span className="text-brand">Theft</span>Guard
        </h1>
      </div>

      <nav className="flex-1 space-y-2">
        {links.map((link) => {
          const Icon = link.icon;
          const isActive = pathname === link.href;

          return (
            <Link
              key={link.href}
              href={link.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                isActive
                  ? "bg-brand/20 text-brand font-medium border border-brand/30 shadow-[0_0_15px_rgba(59,130,246,0.15)]"
                  : "text-foreground/70 hover:bg-glass hover:text-foreground"
              }`}
            >
              <Icon className="w-5 h-5" />
              {link.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto pt-8 pb-4">
        <div className="px-4 py-3 rounded-lg bg-black/20 border border-glass-border">
          <div className="truncate text-sm font-medium">{user.username}</div>
          <div className="mt-1 text-xs text-foreground/50">{user.role === "admin" ? "Yönetici" : user.role === "operator" ? "Operatör" : "İzleyici"}</div>
          <button disabled={loggingOut} type="button" onClick={async () => {
            setLoggingOut(true);
            setError("");
            try { await logout(); } catch (cause) { setError(cause instanceof Error ? cause.message : "Çıkış yapılamadı."); }
            finally { setLoggingOut(false); }
          }} className="mt-3 flex items-center gap-2 text-sm text-foreground/70 hover:text-foreground disabled:opacity-50"><LogOut className="h-4 w-4" />{loggingOut ? "Çıkış yapılıyor…" : "Çıkış yap"}</button>
          {error && <p role="alert" className="mt-2 text-xs text-red-400">{error}</p>}
        </div>
      </div>
    </aside>
  );
}
