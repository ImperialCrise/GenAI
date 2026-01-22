"use client";

import { motion } from "framer-motion";
import Image from "next/image";

interface ImageGridProps {
  images: string[];
  columns?: number;
  className?: string;
  staggerDelay?: number;
}

export default function ImageGrid({
  images,
  columns = 4,
  className = "",
  staggerDelay = 0.1,
}: ImageGridProps) {
  return (
    <div
      className={`grid gap-4 ${className}`}
      style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}
    >
      {images.map((src, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{
            delay: index * staggerDelay,
            duration: 0.4,
            ease: "easeOut",
          }}
          whileHover={{ scale: 1.05, boxShadow: "0 0 30px rgba(0, 240, 255, 0.5)" }}
          className="relative aspect-square rounded-lg overflow-hidden border border-neon-primary/30 bg-neon-bg"
        >
          <Image
            src={src}
            alt={`Generated sample ${index + 1}`}
            fill
            className="object-cover"
          />
        </motion.div>
      ))}
    </div>
  );
}
