"use client";

import { motion } from "framer-motion";
import ParticleBackground from "@/components/ParticleBackground";
import FormulaBlock from "@/components/FormulaBlock";
import NeonCard from "@/components/NeonCard";

export default function Slide06Reverse() {
  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-7xl font-display font-bold text-neon-secondary mb-4"
      >
        Reverse Process
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="relative z-10 text-2xl text-neon-muted mb-12"
      >
        Processus de débruitage p(xₜ₋₁|xₜ)
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-2 gap-8">
        <div className="space-y-6">
          <NeonCard delay={0.4} glow="magenta">
            <h3 className="text-neon-secondary font-display font-bold mb-4">
              Distribution apprise
            </h3>
            <FormulaBlock
              formula="p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})"
              delay={0.6}
            />
          </NeonCard>

          <NeonCard delay={0.8} glow="cyan">
            <h3 className="text-neon-primary font-display font-bold mb-4">
              Calcul de μ_θ
            </h3>
            <FormulaBlock
              formula="\mu_\theta = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)"
              delay={1}
            />
          </NeonCard>
        </div>

        <div className="space-y-6">
          <NeonCard delay={1.2} glow="violet">
            <h3 className="text-neon-accent font-display font-bold mb-4">
              Algorithme de sampling
            </h3>
            <div className="font-mono text-sm space-y-2">
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.4 }}
                className="text-neon-muted"
              >
                <span className="text-neon-primary">1.</span> x_T ~ N(0, I)
              </motion.div>
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.6 }}
                className="text-neon-muted"
              >
                <span className="text-neon-primary">2.</span> for t = T...1:
              </motion.div>
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1.8 }}
                className="text-neon-muted pl-6"
              >
                ε̂ = ε_θ(x_t, t)
              </motion.div>
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 2 }}
                className="text-neon-muted pl-6"
              >
                x_{"{t-1}"} = μ_θ + σ_t · z
              </motion.div>
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 2.2 }}
                className="text-neon-muted"
              >
                <span className="text-neon-primary">3.</span> return x_0
              </motion.div>
            </div>
          </NeonCard>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 2.4 }}
            className="p-4 bg-neon-secondary/10 border border-neon-secondary/30 rounded-xl text-center"
          >
            <p className="text-neon-secondary font-display font-bold">1000 étapes</p>
            <p className="text-neon-muted text-sm">de débruitage itératif</p>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
