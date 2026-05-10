import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ResQAI",
  description: "AI-native multimodal disaster intelligence workspace",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased dark">
      <body className="min-h-full flex flex-col bg-black text-white">{children}</body>
    </html>
  );
}
