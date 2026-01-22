"use client";

import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { useEffect, useState } from "react";

interface AnimatedCounterProps {
  value: number;
  duration?: number;
  delay?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
}

export default function AnimatedCounter({
  value,
  duration = 2,
  delay = 0,
  decimals = 3,
  prefix = "",
  suffix = "",
  className = "",
}: AnimatedCounterProps) {
  const count = useMotionValue(0);
  const rounded = useTransform(count, (v) => v.toFixed(decimals));
  const [displayValue, setDisplayValue] = useState("0.000");

  useEffect(() => {
    const timeout = setTimeout(() => {
      const controls = animate(count, value, {
        duration,
        ease: "easeOut",
        onUpdate: (v) => setDisplayValue(v.toFixed(decimals)),
      });

      return () => controls.stop();
    }, delay * 1000);

    return () => clearTimeout(timeout);
  }, [count, value, duration, delay, decimals]);

  return (
    <motion.span
      className={`font-mono tabular-nums ${className}`}
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, duration: 0.5 }}
    >
      {prefix}
      <motion.span>{displayValue}</motion.span>
      {suffix}
    </motion.span>
  );
}
