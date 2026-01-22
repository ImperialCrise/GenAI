export interface SlideConfig {
  id: number;
  title: string;
  subtitle?: string;
  bgIntensity?: number;
  bgColor?: string;
}

export const SLIDES: SlideConfig[] = [
  { id: 1, title: "DDPM", subtitle: "Denoising Diffusion Probabilistic Models", bgIntensity: 150, bgColor: "#00f0ff" },
  { id: 2, title: "Motivation", subtitle: "Pourquoi les modèles génératifs ?", bgIntensity: 60, bgColor: "#8b5cf6" },
  { id: 3, title: "Vue d'ensemble", subtitle: "Forward & Reverse Process", bgIntensity: 80, bgColor: "#00f0ff" },
  { id: 4, title: "Forward Process", subtitle: "q(xₜ|x₀)", bgIntensity: 70, bgColor: "#00f0ff" },
  { id: 5, title: "Architecture", subtitle: "U-Net ε_θ(xₜ, t)", bgIntensity: 50, bgColor: "#ff00aa" },
  { id: 6, title: "Reverse Process", subtitle: "p(xₜ₋₁|xₜ)", bgIntensity: 70, bgColor: "#ff00aa" },
  { id: 7, title: "Entraînement", subtitle: "L_simple Loss", bgIntensity: 60, bgColor: "#8b5cf6" },
  { id: 8, title: "Visualisation", subtitle: "Forward Noising", bgIntensity: 100, bgColor: "#00f0ff" },
  { id: 9, title: "Visualisation", subtitle: "Reverse Denoising", bgIntensity: 100, bgColor: "#ff00aa" },
  { id: 10, title: "Résultats", subtitle: "Métriques & Galerie", bgIntensity: 80, bgColor: "#00f0ff" },
  { id: 11, title: "Comparaison", subtitle: "Generated vs Real", bgIntensity: 60, bgColor: "#8b5cf6" },
  { id: 12, title: "Merci", subtitle: "Questions ?", bgIntensity: 200, bgColor: "#ff00aa" },
];

export const TOTAL_SLIDES = SLIDES.length;

export const TEAM_MEMBERS = [
  "Marc Guillemot",
  "Rayan Drissi",
  "Emre Ulusoy",
  "Max Nagaishi",
  "Paul Abi-Saad",
];
