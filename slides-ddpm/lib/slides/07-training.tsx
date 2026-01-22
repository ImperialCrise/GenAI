"use client";

import { motion } from "framer-motion";
import ParticleBackground from "@/components/ParticleBackground";
import FormulaBlock from "@/components/FormulaBlock";
import NeonCard from "@/components/NeonCard";

export default function Slide07Training() {
  const trainingSteps = [
    { step: "1", text: "Échantillonner x₀ ~ q(x₀)", color: "cyan" },
    { step: "2", text: "Échantillonner t ~ Uniform(1, T)", color: "cyan" },
    { step: "3", text: "Échantillonner ε ~ N(0, I)", color: "cyan" },
    { step: "4", text: "Calculer x_t via reparameterization", color: "magenta" },
    { step: "5", text: "Prédire ε̂ = ε_θ(x_t, t)", color: "magenta" },
    { step: "6", text: "Backprop sur ||ε - ε̂||²", color: "violet" },
  ];

  const hyperparams = [
    { label: "Epochs", value: "20" },
    { label: "Batch size", value: "128" },
    { label: "Learning rate", value: "2e-4" },
    { label: "Optimizer", value: "Adam" },
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-7xl font-display font-bold text-neon-accent mb-4"
      >
        Entraînement
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="relative z-10 text-2xl text-neon-muted mb-10"
      >
        Fonction de perte simple L_simple
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-2 gap-8">
        <div className="space-y-6">
          <NeonCard delay={0.3} glow="violet">
            <h3 className="text-neon-accent font-display font-bold mb-4">
              Objective
            </h3>
            <FormulaBlock
              formula="L_{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]"
              delay={0.5}
            />
            <p className="text-neon-muted text-sm mt-4">
              Le modèle apprend à prédire le bruit ajouté
            </p>
          </NeonCard>

          <NeonCard delay={0.7} glow="cyan">
            <h3 className="text-neon-primary font-display font-bold mb-4">
              Hyperparamètres
            </h3>
            <div className="grid grid-cols-2 gap-4">
              {hyperparams.map((hp, i) => (
                <motion.div
                  key={hp.label}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.9 + i * 0.1 }}
                  className="text-center p-3 bg-neon-bg/50 rounded-lg border border-neon-primary/20"
                >
                  <p className="text-neon-muted text-xs uppercase">{hp.label}</p>
                  <p className="text-neon-primary font-mono font-bold">{hp.value}</p>
                </motion.div>
              ))}
            </div>
          </NeonCard>
        </div>

        <NeonCard delay={0.4} glow="magenta" className="h-fit">
          <h3 className="text-neon-secondary font-display font-bold mb-4">
            Boucle d&apos;entraînement
          </h3>
          <div className="space-y-3">
            {trainingSteps.map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + i * 0.15 }}
                className="flex items-center gap-3"
              >
                <span
                  className={`
                    w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                    ${step.color === "cyan" ? "bg-neon-primary/20 text-neon-primary" : ""}
                    ${step.color === "magenta" ? "bg-neon-secondary/20 text-neon-secondary" : ""}
                    ${step.color === "violet" ? "bg-neon-accent/20 text-neon-accent" : ""}
                  `}
                >
                  {step.step}
                </span>
                <span className="text-neon-text text-sm font-mono">{step.text}</span>
              </motion.div>
            ))}
          </div>
        </NeonCard>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 2 }}
        className="relative z-10 mt-6 p-4 bg-neon-accent/10 border border-neon-accent/30 rounded-xl text-center"
      >
        <span className="text-neon-accent font-display">
          ~15 minutes d&apos;entraînement sur GPU
        </span>
      </motion.div>
    </div>
  );
}
