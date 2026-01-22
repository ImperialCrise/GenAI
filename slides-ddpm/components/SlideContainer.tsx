"use client";

import { useEffect, useCallback, ReactNode } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { TOTAL_SLIDES } from "@/lib/slides-config";

interface SlideContainerProps {
  children: ReactNode;
  slideNumber: number;
}

export default function SlideContainer({ children, slideNumber }: SlideContainerProps) {
  const router = useRouter();

  const navigate = useCallback(
    (direction: "prev" | "next") => {
      if (direction === "next" && slideNumber < TOTAL_SLIDES) {
        router.push(`/slides/${slideNumber + 1}`);
      } else if (direction === "prev" && slideNumber > 1) {
        router.push(`/slides/${slideNumber - 1}`);
      }
    },
    [router, slideNumber]
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " ") {
        e.preventDefault();
        navigate("next");
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        navigate("prev");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [navigate]);

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-neon-bg">
      <AnimatePresence mode="wait">
        <motion.div
          key={slideNumber}
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -50 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="w-full h-full flex flex-col"
        >
          {children}
        </motion.div>
      </AnimatePresence>

      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4">
        <div className="flex gap-2">
          {Array.from({ length: TOTAL_SLIDES }, (_, i) => (
            <motion.div
              key={i}
              className={`w-2 h-2 rounded-full transition-all duration-300 ${
                i + 1 === slideNumber
                  ? "bg-neon-primary shadow-neon-cyan scale-125"
                  : i + 1 < slideNumber
                  ? "bg-neon-primary/50"
                  : "bg-neon-muted/30"
              }`}
              whileHover={{ scale: 1.5 }}
            />
          ))}
        </div>
        <span className="text-neon-muted text-sm font-mono">
          {slideNumber} / {TOTAL_SLIDES}
        </span>
      </div>

      <div className="absolute bottom-6 right-6 text-neon-muted/50 text-xs">
        ← → pour naviguer
      </div>
    </div>
  );
}
