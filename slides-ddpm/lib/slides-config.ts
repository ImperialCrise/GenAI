export interface SlideConfig {
  id: number;
  title: string;
  subtitle?: string;
}

export const SLIDES: SlideConfig[] = [
  { id: 1, title: "DDPM", subtitle: "Denoising Diffusion Probabilistic Models" },
  { id: 2, title: "Motivation", subtitle: "Pourquoi les modèles génératifs ?" },
  { id: 3, title: "Vue d'ensemble", subtitle: "Forward & Reverse Process" },
  { id: 4, title: "Forward Process", subtitle: "q(xₜ|x₀)" },
  { id: 5, title: "Architecture", subtitle: "U-Net ε_θ(xₜ, t)" },
  { id: 6, title: "Reverse Process", subtitle: "p(xₜ₋₁|xₜ)" },
  { id: 7, title: "Entraînement", subtitle: "L_simple Loss" },
  { id: 8, title: "Visualisation", subtitle: "Forward Noising" },
  { id: 9, title: "Visualisation", subtitle: "Reverse Denoising" },
  { id: 10, title: "Résultats", subtitle: "Métriques & Galerie" },
  { id: 11, title: "Comparaison", subtitle: "Generated vs Real" },
  { id: 12, title: "Merci", subtitle: "Questions ?" },
];

export const TOTAL_SLIDES = SLIDES.length;

export const TEAM_MEMBERS = [
  "Marc Guillemot",
  "Rayan Drissi",
  "Emre Ulusoy",
  "Max Nagaishi",
  "Paul Abi-Saad",
];
