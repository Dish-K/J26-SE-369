import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const archivo = localFont({
  src: "./fonts/archivo-latin.woff2",
  variable: "--font-archivo",
  display: "swap",
  weight: "400 800",
});

export const metadata: Metadata = {
  title: {
    default: "CodeTrace — integrity for technical interviews",
    template: "%s | CodeTrace",
  },
  description:
    "CodeTrace is a research project exploring behavioral signals and human review in live technical interviews.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={archivo.variable}>
      <body>{children}</body>
    </html>
  );
}
