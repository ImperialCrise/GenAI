"use client";

import { motion } from "framer-motion";
import ParticleBackground from "@/components/ParticleBackground";
import UNetDiagram from "@/components/UNetDiagram";
import NeonCard from "@/components/NeonCard";

export default function Slide05UNet() {
  const specs = [
    { label: "Input", value: "28×28×1" },
    { label: "Base channels", value: "128" },
    { label: "Time embedding", value: "128 dim" },
    { label: "Parameters", value: "~8.5M" },
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-4xl font-display font-bold text-neon-primary mb-1"
      >
        Architecture U-Net
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="relative z-10 text-neon-muted mb-4"
      >
        Réseau ε_θ(xₜ, t) pour prédire le bruit
      </motion.p>

      <div className="relative z-10 flex-1 flex flex-col min-h-0">
        <div className="flex-1 min-h-0">
          <UNetDiagram />
        </div>

        <div className="mt-4 grid grid-cols-4 gap-4">
          {specs.map((spec, i) => (
            <NeonCard key={spec.label} delay={2.5 + i * 0.1} glow="cyan" className="py-3 px-4">
              <div className="text-center">
                <p className="text-neon-muted text-[10px] uppercase tracking-wide">{spec.label}</p>
                <p className="text-neon-primary font-mono text-base font-bold mt-0.5">{spec.value}</p>
              </div>
            </NeonCard>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 3 }}
          className="mt-4 grid grid-cols-3 gap-6 text-sm"
        >
          <div className="flex items-start gap-2">
            <span className="text-neon-primary mt-1">▸</span>
            <div>
              <span className="text-neon-text font-bold">TimeEmbedding</span>
              <p className="text-neon-muted text-xs">Sinusoidal positional encoding pour t</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-secondary mt-1">▸</span>
            <div>
              <span className="text-neon-text font-bold">Skip Connections</span>
              <p className="text-neon-muted text-xs">Préservent les détails haute fréquence</p>
            </div>
          </div>
          <div className="flex items-start gap-2">
            <span className="text-neon-accent mt-1">▸</span>
            <div>
              <span className="text-neon-text font-bold">BatchNorm + ReLU</span>
              <p className="text-neon-muted text-xs">Normalisation et activation</p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
