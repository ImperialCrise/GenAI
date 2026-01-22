"use client";

import { motion } from "framer-motion";
import ParticleBackground from "@/components/ParticleBackground";
import FormulaBlock from "@/components/FormulaBlock";
import NeonCard from "@/components/NeonCard";

export default function Slide04Forward() {
  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-7xl font-display font-bold text-neon-primary mb-4"
      >
        Forward Process
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="relative z-10 text-2xl text-neon-muted mb-12"
      >
        Processus de bruitage progressif q(xₜ|x₀)
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-2 gap-8">
        <div className="space-y-6">
          <NeonCard delay={0.4} glow="cyan">
            <h3 className="text-neon-primary font-display font-bold mb-4">
              Distribution conditionnelle
            </h3>
            <FormulaBlock
              formula="q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\mathbf{I})"
              delay={0.6}
            />
          </NeonCard>

          <NeonCard delay={0.8} glow="magenta">
            <h3 className="text-neon-secondary font-display font-bold mb-4">
              Reparameterization Trick
            </h3>
            <FormulaBlock
              formula="x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon"
              delay={1}
            />
            <p className="text-neon-muted text-sm mt-3">
              où ε ~ N(0, I)
            </p>
          </NeonCard>
        </div>

        <div className="space-y-6">
          <NeonCard delay={1.2} glow="violet">
            <h3 className="text-neon-accent font-display font-bold mb-4">
              Schedule linéaire des β
            </h3>
            <div className="space-y-3 font-mono text-sm">
              <div className="flex justify-between">
                <span className="text-neon-muted">β₁ =</span>
                <span className="text-neon-primary">0.0001</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neon-muted">βₜ =</span>
                <span className="text-neon-primary">0.02</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neon-muted">T =</span>
                <span className="text-neon-primary">1000 steps</span>
              </div>
            </div>
          </NeonCard>

          <NeonCard delay={1.4} glow="cyan">
            <h3 className="text-neon-primary font-display font-bold mb-4">
              Définitions
            </h3>
            <div className="space-y-2 font-mono text-sm">
              <FormulaBlock 
                formula="\alpha_t = 1 - \beta_t" 
                delay={1.6}
                displayMode={false}
              />
              <FormulaBlock 
                formula="\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s" 
                delay={1.8}
                displayMode={false}
              />
            </div>
          </NeonCard>
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 2 }}
        className="relative z-10 mt-6 text-center"
      >
        <span className="text-neon-muted text-sm">
          À t=T, x_T ≈ N(0, I) → bruit pur gaussien
        </span>
      </motion.div>
    </div>
  );
}
