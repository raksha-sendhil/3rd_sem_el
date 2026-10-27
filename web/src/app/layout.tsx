import type { Metadata } from "next";
import { Big_Shoulders, Fragment_Mono, Schibsted_Grotesk } from "next/font/google";
import "./globals.css";

const display = Big_Shoulders({
  subsets: ["latin"],
  variable: "--font-display",
  adjustFontFallback: false,
});
const text = Schibsted_Grotesk({ subsets: ["latin"], variable: "--font-text" });
const mono = Fragment_Mono({ subsets: ["latin"], weight: "400", variable: "--font-mono" });

export const metadata: Metadata = {
  title: "sanitise",
  description: "Standard RAG and protected RAG, side by side",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${text.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
