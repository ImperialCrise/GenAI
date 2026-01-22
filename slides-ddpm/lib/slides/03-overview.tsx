"use client";

import { motion } from "framer-motion";
import ParticleBackground from "@/components/ParticleBackground";

export default function Slide03Overview() {
  return (
    <div className="relative flex-1 flex flex-col p-12 overflow-hidden">
      <ParticleBackground intensity={80} color="#00f0ff" />

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-4xl font-display font-bold text-neon-primary mb-8"
      >
        Vue d&apos;ensemble DDPM
      </motion.h2>

      <div className="relative z-10 flex-1 flex items-center justify-center">
        <svg className="w-full max-w-4xl h-80" viewBox="0 0 800 300">
          <motion.rect
            x="50" y="100" width="120" height="100" rx="10"
            fill="#0a0a0f" stroke="#00f0ff" strokeWidth="3"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3, type: "spring" }}
          />
          <motion.text
            x="110" y="145" textAnchor="middle" fill="#e0e0e0" fontSize="14" fontFamily="JetBrains Mono"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            x₀
          </motion.text>
          <motion.text
            x="110" y="170" textAnchor="middle" fill="#6b7280" fontSize="12" fontFamily="JetBrains Mono"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            Image
          </motion.text>

          {[1, 2, 3].map((i) => (
            <motion.g key={i}>
              <motion.line
                x1={170 + (i - 1) * 150} y1="150"
                x2={200 + (i - 1) * 150} y2="150"
                stroke="#00f0ff" strokeWidth="2"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ delay: 0.6 + i * 0.2, duration: 0.4 }}
              />
              <motion.polygon
                points={`${200 + (i - 1) * 150},145 ${215 + (i - 1) * 150},150 ${200 + (i - 1) * 150},155`}
                fill="#00f0ff"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 + i * 0.2 }}
              />
              <motion.text
                x={185 + (i - 1) * 150} y="135" textAnchor="middle" fill="#00f0ff" fontSize="10"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 + i * 0.2 }}
              >
                +ε
              </motion.text>
            </motion.g>
          ))}

          {[1, 2, 3].map((i) => (
            <motion.g key={`box-${i}`}>
              <motion.rect
                x={220 + (i - 1) * 150} y="100" width="120" height="100" rx="10"
                fill="#0a0a0f" stroke="#8b5cf6" strokeWidth="2"
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.8 + i * 0.2, type: "spring" }}
              />
              <motion.text
                x={280 + (i - 1) * 150} y="145" textAnchor="middle" fill="#e0e0e0" fontSize="14"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1 + i * 0.2 }}
              >
                x{i === 3 ? "ᵧ" : `${i}`}
              </motion.text>
              <motion.text
                x={280 + (i - 1) * 150} y="170" textAnchor="middle" fill="#6b7280" fontSize="10"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1 + i * 0.2 }}
              >
                {i === 1 ? "Peu bruité" : i === 2 ? "Plus bruité" : "Bruit pur"}
              </motion.text>
            </motion.g>
          ))}

          <motion.text
            x="400" y="50" textAnchor="middle" fill="#00f0ff" fontSize="16" fontWeight="bold"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.8 }}
          >
            Forward Process q(xₜ|xₜ₋₁) → Ajout de bruit
          </motion.text>

          {[1, 2, 3].map((i) => (
            <motion.g key={`rev-${i}`}>
              <motion.line
                x1={640 - (i - 1) * 150} y1="230"
                x2={610 - (i - 1) * 150} y2="230"
                stroke="#ff00aa" strokeWidth="2"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ delay: 2 + i * 0.2, duration: 0.4 }}
              />
              <motion.polygon
                points={`${610 - (i - 1) * 150},225 ${595 - (i - 1) * 150},230 ${610 - (i - 1) * 150},235`}
                fill="#ff00aa"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 2.2 + i * 0.2 }}
              />
              <motion.text
                x={625 - (i - 1) * 150} y="250" textAnchor="middle" fill="#ff00aa" fontSize="10"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 2.2 + i * 0.2 }}
              >
                -ε̂
              </motion.text>
            </motion.g>
          ))}

          <motion.text
            x="400" y="280" textAnchor="middle" fill="#ff00aa" fontSize="16" fontWeight="bold"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 2.8 }}
          >
            Reverse Process p(xₜ₋₁|xₜ) → Débruitage avec U-Net
          </motion.text>
        </svg>
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 3 }}
        className="relative z-10 text-center text-neon-muted text-sm"
      >
        Le modèle apprend à inverser le processus de diffusion
      </motion.div>
    </div>
  );
}
