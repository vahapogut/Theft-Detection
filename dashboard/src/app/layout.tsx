import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import AuthProvider from "@/components/AuthProvider";

export const metadata: Metadata = {
  title: "TheftGuard Dashboard",
  description: "Advanced Anti-Theft AI System Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="tr">
      <body className="antialiased flex">
        <AuthProvider>
          <Sidebar />
          <main className="flex-1 p-6 h-screen overflow-y-auto">
            {children}
          </main>
        </AuthProvider>
      </body>
    </html>
  );
}
