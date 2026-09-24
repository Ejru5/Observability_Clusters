import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Observability Clusters v2",
  description: "Trace-level, LLM-first error clustering dashboard for Mistral AI pipelines.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" style={{ colorScheme: "dark" }}>
      <body>{children}</body>
    </html>
  );
}
