"use client";

import { motion } from "framer-motion";
import { Fragment } from "react";
import ParticleBackground from "@/components/ParticleBackground";
import NeonCard from "@/components/NeonCard";

export default function Slide09VizReverse() {
  const timesteps = [999, 800, 600, 400, 200, 0];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-7xl font-display font-bold text-neon-secondary mb-4"
      >
        Visualisation Reverse
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="relative z-10 text-2xl text-neon-muted mb-8"
      >
        Processus de débruitage : du bruit pur aux chiffres générés
      </motion.p>

      <div className="relative z-10 flex-1 flex flex-col justify-center min-h-0">
        <NeonCard delay={0.3} glow="magenta" className="p-6 flex-1 min-h-0 flex flex-col">
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
                  className="flex items-center justify-end pr-4 text-neon-secondary font-mono text-xs"
                >
                  t={t}
                </motion.div>
                {[1, 2, 3, 4, 5, 6].map((sample) => (
                  <motion.div
                    key={`${t}-${sample}`}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.5 + rowIdx * 0.1 + sample * 0.03 }}
                    whileHover={{ scale: 1.1, boxShadow: "0 0 30px rgba(255, 0, 170, 0.5)" }}
                    className="aspect-square bg-neon-bg border border-neon-secondary/30 rounded overflow-hidden relative cursor-pointer max-h-[100px] w-full mx-auto"
                    style={{
                      background: t === 0 
                        ? `linear-gradient(135deg, rgba(10,10,15,0.9) 0%, rgba(30,30,40,0.9) 100%)`
                        : `linear-gradient(135deg, 
                            rgba(255,0,170,${0.1 + ((999 - t) / 1000) * 0.1}) 0%, 
                            rgba(139,92,246,${0.1 + ((999 - t) / 1000) * 0.1}) 100%)`,
                    }}
                  >
                    <div
                      className="absolute inset-0 flex items-center justify-center text-xl font-bold"
                      style={{ 
                        opacity: t === 0 ? 1 : (999 - t) / 1200,
                        color: "#e0e0e0",
                      }}
                    >
                      {t === 0 ? ["3", "7", "1", "8", "2", "5"][sample - 1] : ""}
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
            <div className="w-8 h-2 bg-gradient-to-r from-neon-secondary/50 to-neon-accent/50 rounded" />
            <span className="text-neon-muted text-sm">Bruit pur (t=999)</span>
          </div>
          <span className="text-neon-muted">→</span>
          <div className="flex items-center gap-2">
            <div className="w-8 h-2 bg-gradient-to-r from-neon-accent/50 to-neon-text rounded" />
            <span className="text-neon-muted text-sm">Image générée (t=0)</span>
          </div>
        </motion.div>
      </div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2 }}
        className="relative z-10 text-center text-neon-muted text-sm mt-4"
      >
        💡 Remplacez ce placeholder par le GIF <code className="text-neon-secondary">reverse_denoising.gif</code> exporté du notebook
      </motion.p>
    </div>
  );
}
