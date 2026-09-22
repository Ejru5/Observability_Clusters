import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cluster Observatory — Mistral Span Clustering",
  description:
    "Two-level embedding-based clustering dashboard for Mistral Observability spans. Macro + micro cluster analysis powered by UMAP + HDBSCAN + mistral-large.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body style={{ fontFamily: "'Plus Jakarta Sans', system-ui, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
