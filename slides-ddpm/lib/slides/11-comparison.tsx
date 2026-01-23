"use client";

import { motion } from "framer-motion";
import ParticleBackground from "@/components/ParticleBackground";
import NeonCard from "@/components/NeonCard";
import AnimatedBar from "@/components/AnimatedBar";

export default function Slide11Comparison() {
  const classDistribution = [
    { digit: 0, count: 97 },
    { digit: 1, count: 32 },
    { digit: 2, count: 163 },
    { digit: 3, count: 128 },
    { digit: 4, count: 58 },
    { digit: 5, count: 143 },
    { digit: 6, count: 79 },
    { digit: 7, count: 74 },
    { digit: 8, count: 128 },
    { digit: 9, count: 95 },
  ];

  const maxCount = Math.max(...classDistribution.map((d) => d.count));

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-4xl font-display font-bold text-neon-accent mb-1"
      >
        Comparaison
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="relative z-10 text-neon-muted mb-4"
      >
        Generated vs Real : distribution de confiance et de classes
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-2 gap-6 min-h-0 overflow-hidden">
        <NeonCard delay={0.3} glow="cyan" className="flex flex-col min-h-0">
          <h3 className="text-neon-primary font-display font-bold mb-4 text-sm uppercase">
            Distribution de confiance
          </h3>
          
          <div className="flex-1 min-h-0 relative flex items-end justify-around px-4">
            {[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((conf, i) => {
              const genHeight = conf < 0.8 ? 5 + Math.random() * 10 : (conf === 1.0 ? 90 : 30 + (conf - 0.8) * 200);
              const realHeight = conf < 0.95 ? 2 + Math.random() * 5 : 95;
              
              return (
                <div key={conf} className="flex flex-col items-center gap-1 flex-1 h-full justify-end">
                  <div className="relative w-full flex justify-center gap-1 h-full items-end pb-1">
                    <motion.div
                      className="w-2.5 bg-neon-primary rounded-t"
                      initial={{ height: 0 }}
                      animate={{ height: `${Math.min(genHeight, 100)}%` }}
                      transition={{ delay: 0.5 + i * 0.1, duration: 0.8, ease: "easeOut" }}
                      style={{ originY: 1 }}
                    />
                    <motion.div
                      className="w-2.5 bg-neon-secondary rounded-t opacity-60"
                      initial={{ height: 0 }}
                      animate={{ height: `${Math.min(realHeight, 100)}%` }}
                      transition={{ delay: 0.6 + i * 0.1, duration: 0.8, ease: "easeOut" }}
                      style={{ originY: 1 }}
                    />
                  </div>
                  <span className="text-neon-muted text-[10px]">{conf}</span>
                </div>
              );
            })}
          </div>

          <div className="flex justify-center gap-6 mt-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-neon-primary" />
              <span className="text-neon-text text-xs">Generated (μ=0.941)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-neon-secondary opacity-60" />
              <span className="text-neon-text text-xs">Real (μ=0.993)</span>
            </div>
          </div>
        </NeonCard>

        <NeonCard delay={0.4} glow="magenta" className="flex flex-col min-h-0 overflow-hidden">
          <h3 className="text-neon-secondary font-display font-bold mb-3 text-sm uppercase">
            Distribution des classes générées
          </h3>
          
          <div className="flex-1 min-h-0 flex flex-col justify-between space-y-0.5 pr-2">
            {classDistribution.map((item, i) => (
              <AnimatedBar
                key={item.digit}
                label={`${item.digit}`}
                value={item.count}
                maxValue={maxCount}
                delay={0.6 + i * 0.08}
                color={item.count > 100 ? "#ff00aa" : "#00f0ff"}
              />
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2 }}
            className="mt-3 p-2 bg-neon-accent/10 border border-neon-accent/30 rounded-lg"
          >
            <p className="text-neon-muted text-[10px] text-center italic">
              Biais vers les chiffres 2, 5 et 8
            </p>
          </motion.div>
        </NeonCard>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 2.5 }}
        className="relative z-10 mt-4 grid grid-cols-3 gap-4"
      >
        <div className="text-center p-2 bg-neon-primary/10 border border-neon-primary/30 rounded-lg">
          <span className="text-neon-primary font-display font-bold text-lg">94.1%</span>
          <p className="text-neon-muted text-[10px] uppercase">Confiance Gen</p>
        </div>
        <div className="text-center p-2 bg-neon-secondary/10 border border-neon-secondary/30 rounded-lg">
          <span className="text-neon-secondary font-display font-bold text-lg">99.3%</span>
          <p className="text-neon-muted text-[10px] uppercase">Confiance Réelle</p>
        </div>
        <div className="text-center p-2 bg-neon-accent/10 border border-neon-accent/30 rounded-lg">
          <span className="text-neon-accent font-display font-bold text-lg">-5.2%</span>
          <p className="text-neon-muted text-[10px] uppercase">Écart</p>
        </div>
      </motion.div>
    </div>
  );
}
