import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        neon: {
          bg: "#0a0a0f",
          primary: "#00f0ff",
          secondary: "#ff00aa",
          accent: "#8b5cf6",
          text: "#e0e0e0",
          muted: "#6b7280",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        display: ["Orbitron", "sans-serif"],
      },
      boxShadow: {
        "neon-cyan": "0 0 20px rgba(0, 240, 255, 0.5), 0 0 40px rgba(0, 240, 255, 0.3)",
        "neon-magenta": "0 0 20px rgba(255, 0, 170, 0.5), 0 0 40px rgba(255, 0, 170, 0.3)",
        "neon-violet": "0 0 20px rgba(139, 92, 246, 0.5), 0 0 40px rgba(139, 92, 246, 0.3)",
        "neon-glow": "0 0 30px rgba(0, 240, 255, 0.4), 0 0 60px rgba(255, 0, 170, 0.2)",
      },
      animation: {
        "pulse-neon": "pulseNeon 2s ease-in-out infinite",
        "glitch": "glitch 0.5s ease-in-out",
        "float": "float 3s ease-in-out infinite",
        "scan": "scan 2s linear infinite",
      },
      keyframes: {
        pulseNeon: {
          "0%, 100%": { opacity: "1", filter: "brightness(1)" },
          "50%": { opacity: "0.8", filter: "brightness(1.3)" },
        },
        glitch: {
          "0%": { transform: "translate(0)" },
          "20%": { transform: "translate(-2px, 2px)" },
          "40%": { transform: "translate(-2px, -2px)" },
          "60%": { transform: "translate(2px, 2px)" },
          "80%": { transform: "translate(2px, -2px)" },
          "100%": { transform: "translate(0)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" },
        },
        scan: {
          "0%": { backgroundPosition: "0% 0%" },
          "100%": { backgroundPosition: "0% 100%" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
