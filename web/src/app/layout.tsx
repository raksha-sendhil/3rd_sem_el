import type { Metadata } from "next";
import { Bricolage_Grotesque, Pixelify_Sans, Silkscreen, VT323 } from "next/font/google";
import "./globals.css";

const pixel = Pixelify_Sans({ subsets: ["latin"], variable: "--font-pixel" });
const label = Silkscreen({ subsets: ["latin"], weight: "400", variable: "--font-label" });
const text = Bricolage_Grotesque({ subsets: ["latin"], variable: "--font-text" });
const term = VT323({ subsets: ["latin"], weight: "400", variable: "--font-term" });

export const metadata: Metadata = {
  title: "sanitise",
  description: "Standard RAG and protected RAG, side by side",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const fonts = [pixel, label, text, term].map((f) => f.variable).join(" ");
  return (
    <html lang="en" className={fonts}>
      <body>{children}</body>
    </html>
  );
}
