"use client";

import * as Tooltip from "@radix-ui/react-tooltip";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { setRequestDomain } from "@/api/client";
import { DEFAULT_DOMAIN, type Domain, type DomainId, domainById } from "@/lib/domains";

const STORAGE_KEY = "sanitise.domain"; // the same key as the legal branch's dashboard

type DomainState = { domain: Domain; setDomain: (id: DomainId) => void };

const DomainContext = createContext<DomainState>({
  domain: domainById(DEFAULT_DOMAIN),
  setDomain: () => {},
});

/** The chosen use case (health, legal, finance), remembered in this browser. Only a ready
 * domain can be chosen; a remembered one that is no longer ready falls back to the default. */
function DomainProvider({ children }: { children: ReactNode }) {
  const [id, setId] = useState<DomainId>(DEFAULT_DOMAIN);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY) as DomainId | null;
      if (saved && domainById(saved).id === saved && domainById(saved).ready) setId(saved);
    } catch {
      // private window or blocked storage: keep the default
    }
  }, []);

  // set while rendering, not in an effect: the pages' effects run before this provider's and
  // their first requests must already carry the domain
  setRequestDomain(id);

  const setDomain = (next: DomainId) => {
    if (!domainById(next).ready) return;
    setId(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // not remembered, still switched
    }
  };

  return <DomainContext.Provider value={{ domain: domainById(id), setDomain }}>{children}</DomainContext.Provider>;
}

export const useDomain = () => useContext(DomainContext);

export function Providers({ children }: { children: ReactNode }) {
  return (
    <Tooltip.Provider>
      <DomainProvider>{children}</DomainProvider>
    </Tooltip.Provider>
  );
}
