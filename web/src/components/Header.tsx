"use client";

import { FileText, Swords, Workflow } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { API_URL, api } from "@/api/client";

const NAV = [
  { to: "/", label: "Overview", icon: Workflow, color: "#6D4AFF", id: "overview" },
  { to: "/attacks", label: "Attacks", icon: Swords, color: "#FF4D3D", id: "attacks" },
  { to: "/documents", label: "Documents", icon: FileText, color: "#0FB5A0", id: "documents" },
];

export const Logo = () => (
  <div className="flex items-center gap-3" data-testid="brand-logo">
    <svg width="38" height="38" viewBox="0 0 38 38" aria-hidden>
      <rect x="1" y="1" width="17" height="17" rx="5" fill="#FF4D3D" />
      <rect x="20" y="1" width="17" height="17" rx="5" fill="#FFC53D" />
      <rect x="1" y="20" width="17" height="17" rx="5" fill="#2D8CFF" />
      <rect x="20" y="20" width="17" height="17" rx="5" fill="#6D4AFF" />
      <rect x="24" y="26" width="9" height="5" rx="2.5" fill="#fff" />
      <path d="M5 9.5h9" stroke="#fff" strokeWidth="2.6" strokeLinecap="round" />
    </svg>
    <div className="font-display text-lg font-extrabold tracking-tight text-[#17162B]">Sanitise AI</div>
  </div>
);

/** Whether the API answers; checked every 15 s. Only a failure is shown. */
function useOnline(): boolean | null {
  const [online, setOnline] = useState<boolean | null>(null);
  useEffect(() => {
    let alive = true;
    const check = () =>
      api
        .health()
        .then(() => alive && setOnline(true))
        .catch(() => alive && setOnline(false));
    check();
    const timer = setInterval(check, 15000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  return online;
}

export function Header() {
  const path = usePathname();
  const online = useOnline();
  return (
    <header
      data-testid="app-header"
      className="sticky top-0 z-40 border-b border-[#E6E1D6] bg-[#F6F4EF]/80 backdrop-blur-xl"
    >
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center gap-3 px-4 py-3 sm:px-8">
        <Logo />
        <nav className="order-3 flex w-full gap-1 overflow-x-auto rounded-full border border-[#E6E1D6] bg-[#EFECE4] p-1 md:order-none md:ml-6 md:w-auto">
          {NAV.map((n) => {
            const active = path === n.to;
            return (
              <Link
                key={n.to}
                href={n.to}
                data-testid={`nav-${n.id}`}
                className={`flex items-center gap-2 whitespace-nowrap rounded-full px-4 py-2 text-sm font-semibold transition-[background-color,color,transform] duration-200 hover:-translate-y-0.5 ${
                  active ? "text-white shadow-md" : "text-[#4A4760] hover:bg-white"
                }`}
                style={active ? { background: n.color } : undefined}
              >
                <n.icon className="h-4 w-4" strokeWidth={2.4} />
                {n.label}
              </Link>
            );
          })}
        </nav>
        {online === false && (
          <span className="chip ml-auto bg-[#FFE8E4] text-[#B42318]" title={API_URL}>
            <span className="pulse-dot bg-[#FF4D3D]" /> API unreachable
          </span>
        )}
      </div>
    </header>
  );
}
