"use client";

import { motion } from "framer-motion";
import ParticleBackground from "@/components/ParticleBackground";
import NeonCard from "@/components/NeonCard";

export default function Slide02Motivation() {
  const models = [
    { name: "GANs", year: "2014", desc: "Adversarial training", icon: "⚔️" },
    { name: "VAEs", year: "2013", desc: "Latent space encoding", icon: "🔮" },
    { name: "Flow", year: "2015", desc: "Invertible transforms", icon: "🌊" },
    { name: "Diffusion", year: "2020", desc: "Denoising process", icon: "✨" },
  ];

  return (
    <div className="relative flex-1 flex flex-col p-12 overflow-hidden">
      <ParticleBackground intensity={60} color="#8b5cf6" />

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-4xl font-display font-bold text-neon-primary mb-2"
      >
        Pourquoi les modèles génératifs ?
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-neon-muted mb-8"
      >
        Évolution des approches de génération d&apos;images
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-4 gap-6">
        {models.map((model, i) => (
          <NeonCard 
            key={model.name} 
            delay={0.5 + i * 0.2}
            glow={model.name === "Diffusion" ? "magenta" : "cyan"}
            className={model.name === "Diffusion" ? "ring-2 ring-neon-secondary" : ""}
          >
            <div className="text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.8 + i * 0.2, type: "spring" }}
                className="text-5xl mb-4"
              >
                {model.icon}
              </motion.div>
              <h3 className="text-xl font-display font-bold text-neon-text mb-1">
                {model.name}
              </h3>
              <p className="text-neon-primary text-sm mb-2">{model.year}</p>
              <p className="text-neon-muted text-xs">{model.desc}</p>
            </div>
          </NeonCard>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.5 }}
        className="relative z-10 mt-8 p-6 bg-neon-secondary/10 border border-neon-secondary/30 rounded-xl"
      >
        <h3 className="text-neon-secondary font-display font-bold mb-2">
          Avantages de la diffusion
        </h3>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div className="flex items-center gap-2">
            <span className="text-neon-primary">▸</span>
            <span className="text-neon-text">Stabilité d&apos;entraînement</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-neon-primary">▸</span>
            <span className="text-neon-text">Qualité d&apos;échantillons</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-neon-primary">▸</span>
            <span className="text-neon-text">Fondements théoriques solides</span>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
