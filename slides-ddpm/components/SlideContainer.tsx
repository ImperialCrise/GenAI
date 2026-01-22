"use client";

import { useEffect, useCallback, ReactNode, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { TOTAL_SLIDES, SLIDES } from "../lib/slides-config";
import ParticleBackground from "./ParticleBackground";

interface SlideContainerProps {
  children: ReactNode;
  slideNumber: number;
}

export default function SlideContainer({ children, slideNumber }: SlideContainerProps) {
  const router = useRouter();
  const [scale, setScale] = useState(1);
  const currentSlide = SLIDES.find(s => s.id === slideNumber);

  const handleResize = useCallback(() => {
    // Base de conception plus compacte pour forcer un zoom naturel sur le texte
    const targetWidth = 1280;
    const targetHeight = 720;
    const widthScale = window.innerWidth / targetWidth;
    const heightScale = window.innerHeight / targetHeight;
    setScale(Math.min(widthScale, heightScale));
  }, []);

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
    handleResize();
    window.addEventListener("resize", handleResize);
    
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
    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [navigate, handleResize]);

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-neon-bg flex items-center justify-center">
      {/* Background toujours en full screen réel */}
      <ParticleBackground 
        intensity={currentSlide?.bgIntensity || 100} 
        color={currentSlide?.bgColor || "#00f0ff"} 
      />

      {/* Wrapper de scaling intelligent */}
      <div 
        style={{ 
          transform: `scale(${scale})`,
          transformOrigin: "center center",
          width: "1280px",
          height: "720px",
          flexShrink: 0
        }}
        className="relative z-10 overflow-hidden flex flex-col"
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={slideNumber}
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="w-full h-full flex flex-col p-8"
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* UI Navigation - Fixe sur l'écran réel */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4 z-50">
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

      <div className="absolute bottom-6 right-6 text-neon-muted/50 text-xs hidden sm:block">
        ← → pour naviguer
      </div>
    </div>
  );
}

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
