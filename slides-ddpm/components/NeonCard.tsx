"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

interface NeonCardProps {
  children: ReactNode;
  className?: string;
  delay?: number;
  glow?: "cyan" | "magenta" | "violet";
}

export default function NeonCard({
  children,
  className = "",
  delay = 0,
  glow = "cyan",
}: NeonCardProps) {
  const glowColors = {
    cyan: "border-neon-primary/50 shadow-neon-cyan",
    magenta: "border-neon-secondary/50 shadow-neon-magenta",
    violet: "border-neon-accent/50 shadow-neon-violet",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: "easeOut" }}
      className={`
        bg-neon-bg/80 backdrop-blur-md rounded-xl border-2 p-6
        ${glowColors[glow]}
        ${className}
      `}
    >
      {children}
    </motion.div>
  );
}
