"use client";

import { motion } from "framer-motion";
import { Fragment } from "react";
import ParticleBackground from "@/components/ParticleBackground";
import NeonCard from "@/components/NeonCard";

export default function Slide08VizForward() {
  const timesteps = [0, 150, 300, 450, 600, 750, 900];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-7xl font-display font-bold text-neon-primary mb-4"
      >
        Visualisation Forward
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="relative z-10 text-2xl text-neon-muted mb-8"
      >
        Processus de bruitage progressif sur 6 échantillons MNIST
      </motion.p>

      <div className="relative z-10 flex-1 flex flex-col justify-center min-h-0">
        <NeonCard delay={0.3} glow="cyan" className="p-6 flex-1 min-h-0 flex flex-col">
          <div className="grid gap-2 flex-1 min-h-0 overflow-hidden" style={{ gridTemplateColumns: `80px repeat(6, 1fr)` }}>
            <div></div>
            {[1, 2, 3, 4, 5, 6].map((sample) => (
              <motion.div
                key={sample}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 + sample * 0.05 }}
                className="text-center text-neon-muted text-xs"
              >
                S{sample}
              </motion.div>
            ))}

            {timesteps.map((t, rowIdx) => (
              <Fragment key={t}>
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 + rowIdx * 0.1 }}
                  className="flex items-center justify-end pr-4 text-neon-primary font-mono text-xs"
                >
                  t={t}
                </motion.div>
                {[1, 2, 3, 4, 5, 6].map((sample) => (
                  <motion.div
                    key={`${t}-${sample}`}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.5 + rowIdx * 0.1 + sample * 0.03 }}
                    className="aspect-square bg-neon-bg border border-neon-primary/30 rounded overflow-hidden relative max-h-[100px] w-full mx-auto"
                    style={{
                      background: `linear-gradient(135deg, 
                        rgba(0,240,255,${0.1 + (t / 1000) * 0.2}) 0%, 
                        rgba(139,92,246,${0.1 + (t / 1000) * 0.2}) 100%)`,
                    }}
                  >
                    <div
                      className="absolute inset-0 flex items-center justify-center text-neon-text/50 text-[10px] font-mono"
                      style={{ opacity: 1 - t / 1200 }}
                    >
                      {t === 0 ? ["7", "2", "1", "0", "4", "9"][sample - 1] : ""}
                    </div>
                    
                    <div
                      className="absolute inset-0"
                      style={{
                        background: `url("data:image/svg+xml,%3Csvg viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100' height='100' filter='url(%23n)'/%3E%3C/svg%3E")`,
                        opacity: t / 1200,
                        mixBlendMode: "overlay",
                      }}
                    />
                  </motion.div>
                ))}
              </Fragment>
            ))}
          </div>
        </NeonCard>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="mt-6 flex justify-center items-center gap-4"
        >
          <div className="flex items-center gap-2">
            <div className="w-8 h-2 bg-gradient-to-r from-neon-bg to-neon-primary/50 rounded" />
            <span className="text-neon-muted text-sm">Image nette</span>
          </div>
          <span className="text-neon-muted">→</span>
          <div className="flex items-center gap-2">
            <div className="w-8 h-2 bg-gradient-to-r from-neon-primary/50 to-neon-accent/50 rounded" />
            <span className="text-neon-muted text-sm">Bruit pur</span>
          </div>
        </motion.div>
      </div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2 }}
        className="relative z-10 text-center text-neon-muted text-sm mt-4"
      >
        💡 Remplacez ce placeholder par le GIF <code className="text-neon-primary">forward_noising.gif</code> exporté du notebook
      </motion.p>
    </div>
  );
}
