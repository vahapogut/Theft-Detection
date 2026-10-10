const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "/api").replace(/\/+$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE}/${path.replace(/^\/+/, "")}`;
}

export async function apiFetch(path: string, options: RequestInit = {}): Promise<Response> {
  const headers = new Headers(options.headers);
  const method = (options.method || "GET").toUpperCase();
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    headers.set("X-Requested-With", "XMLHttpRequest");
  }
  const response = await fetch(apiUrl(path), {
    ...options,
    headers,
    credentials: "include",
    cache: "no-store",
  });
  if (response.status === 401 && !path.startsWith("/auth/") && typeof window !== "undefined") {
    window.dispatchEvent(new Event("theftguard:unauthorized"));
  }
  return response;
}

export function wsUrl(path: string): string {
  const url = new URL(apiUrl(path), window.location.origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}
