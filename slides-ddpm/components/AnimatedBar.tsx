"use client";

import { motion } from "framer-motion";

interface AnimatedBarProps {
  value: number;
  maxValue: number;
  label: string;
  delay?: number;
  color?: string;
}

export default function AnimatedBar({
  value,
  maxValue,
  label,
  delay = 0,
  color = "#00f0ff",
}: AnimatedBarProps) {
  const percentage = (value / maxValue) * 100;

  return (
    <div className="flex items-center gap-4">
      <span className="w-8 text-right text-neon-text font-mono text-xs">{label}</span>
      <div className="flex-1 h-5 bg-neon-bg border border-neon-primary/30 rounded overflow-hidden">
        <motion.div
          className="h-full rounded"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ delay, duration: 1, ease: "easeOut" }}
        />
      </div>
      <motion.span
        className="w-12 text-neon-text font-mono text-xs"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: delay + 0.5 }}
      >
        {value}
      </motion.span>
    </div>
  );
}
