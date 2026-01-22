import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DDPM - Denoising Diffusion Probabilistic Models",
  description: "Présentation EPITA SCIA - DDPM sur MNIST",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr">
      <body className="antialiased">
        <div className="noise-overlay" />
        {children}
      </body>
    </html>
  );
}
