"use client";

import * as Tooltip from "@radix-ui/react-tooltip";
import type { ReactNode } from "react";
import { Toaster } from "sonner";

export function Providers({ children }: { children: ReactNode }) {
  return (
    <Tooltip.Provider>
      {children}
      <Toaster position="bottom-right" theme="light" richColors={false} />
    </Tooltip.Provider>
  );
}
