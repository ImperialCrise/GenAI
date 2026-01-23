"use client";

import { motion } from "framer-motion";
import { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

interface FormulaBlockProps {
  formula: string;
  delay?: number;
  className?: string;
  displayMode?: boolean;
}

export default function FormulaBlock({
  formula,
  delay = 0,
  className = "",
  displayMode = true,
}: FormulaBlockProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      katex.render(formula, containerRef.current, {
        displayMode,
        throwOnError: false,
        trust: true,
      });
    }
  }, [formula, displayMode]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.6, ease: "easeOut" }}
      className={`${className}`}
    >
      <div
        ref={containerRef}
        className="text-neon-text bg-neon-bg/50 backdrop-blur-sm px-8 py-6 rounded-lg border border-neon-primary/30 shadow-neon-cyan/20 scale-[1.2] origin-center overflow-hidden"
      />
    </motion.div>
  );
}
