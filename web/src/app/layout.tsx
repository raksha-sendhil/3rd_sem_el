import type { Metadata } from "next";
import { Bricolage_Grotesque, Instrument_Sans, JetBrains_Mono } from "next/font/google";
import { Providers } from "@/components/Providers";
import { Header } from "@/components/Header";
import "./globals.css";

const display = Bricolage_Grotesque({
  subsets: ["latin"],
  axes: ["opsz"],
  variable: "--font-display",
});
const body = Instrument_Sans({ subsets: ["latin"], variable: "--font-body" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Sanitise AI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const fonts = [display, body, mono].map((f) => f.variable).join(" ");
  return (
    <html lang="en" className={fonts}>
      <body>
        <Providers>
          <div className="app-bg min-h-screen">
            <Header />
            <main className="mx-auto max-w-[1440px] px-4 pb-24 pt-8 sm:px-8">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
