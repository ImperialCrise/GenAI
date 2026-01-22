"use client";

import { motion } from "framer-motion";
import ParticleBackground from "@/components/ParticleBackground";
import GlitchText from "@/components/GlitchText";
import { TEAM_MEMBERS } from "@/lib/slides-config";

export default function Slide12Merci() {
  const conclusions = [
    "Implémentation complète de DDPM sur MNIST",
    "Forward process q(xₜ|x₀) avec schedule linéaire",
    "U-Net conditionné par le temps (~8.5M params)",
    "Reverse denoising en 1000 steps",
    "Évaluation par classifieur : 94.1% confiance",
  ];

  return (
    <div className="relative flex-1 flex flex-col items-center justify-center overflow-hidden">

      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="relative z-10 text-center mb-8"
      >
        <h1 className="text-7xl font-display font-black mb-4">
          <GlitchText 
            text="Merci !" 
            className="text-neon-secondary text-glow-magenta"
          />
        </h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="text-2xl text-neon-primary font-display"
        >
          Questions ?
        </motion.p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.2 }}
        className="relative z-10 max-w-2xl mb-8"
      >
        <div className="grid grid-cols-1 gap-2 text-sm">
          {conclusions.map((point, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 1.4 + i * 0.15 }}
              className="flex items-center gap-3 px-4 py-2 bg-neon-bg/60 backdrop-blur-sm border border-neon-primary/20 rounded-lg"
            >
              <span className="text-neon-primary">✓</span>
              <span className="text-neon-text">{point}</span>
            </motion.div>
          ))}
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2.5 }}
        className="relative z-10"
      >
        <p className="text-neon-muted text-sm mb-4 text-center">Équipe</p>
        <div className="flex flex-wrap justify-center gap-4">
          {TEAM_MEMBERS.map((member, index) => (
            <motion.div
              key={member}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ 
                delay: 2.7 + index * 0.1,
                type: "spring",
                stiffness: 200,
              }}
              whileHover={{ 
                scale: 1.1, 
                boxShadow: "0 0 30px rgba(255, 0, 170, 0.5)",
              }}
              className="px-6 py-3 bg-neon-bg/80 backdrop-blur-sm border-2 border-neon-secondary/50 rounded-xl cursor-pointer"
            >
              <span className="text-neon-text font-mono font-bold">{member}</span>
            </motion.div>
          ))}
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 3.5 }}
        className="relative z-10 mt-12 flex items-center justify-center gap-6"
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          className="w-16 h-16 rounded-full border-2 border-dashed border-neon-primary/30"
        />
        <div className="text-center">
          <p className="text-neon-accent text-sm">EPITA</p>
          <p className="text-neon-secondary text-lg font-display font-bold">ING3 SCIA</p>
          <p className="text-neon-primary text-sm">Janvier 2026</p>
        </div>
        <motion.div
          animate={{ rotate: -360 }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          className="w-16 h-16 rounded-full border-2 border-dashed border-neon-secondary/30"
        />
      </motion.div>

      <motion.div
        className="absolute inset-0 pointer-events-none"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.1 }}
        transition={{ delay: 1 }}
      >
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 rounded-full bg-neon-secondary"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              scale: [1, 2, 1],
              opacity: [0.3, 1, 0.3],
            }}
            transition={{
              duration: 2 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </motion.div>
    </div>
  );
}
