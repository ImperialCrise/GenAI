"use client";

import { motion } from "framer-motion";

export default function UNetDiagram() {
  const blocks = [
    { id: "input", label: "Input\n28×28×1", x: 50, y: 200, color: "#00f0ff" },
    { id: "down1", label: "Down 1\n14×14×128", x: 150, y: 150, color: "#00f0ff" },
    { id: "down2", label: "Down 2\n7×7×256", x: 250, y: 100, color: "#00f0ff" },
    { id: "bottleneck", label: "Bottleneck\n7×7×512", x: 350, y: 50, color: "#ff00aa" },
    { id: "up1", label: "Up 1\n7×7×256", x: 450, y: 100, color: "#8b5cf6" },
    { id: "up2", label: "Up 2\n14×14×128", x: 550, y: 150, color: "#8b5cf6" },
    { id: "output", label: "Output\n28×28×1", x: 650, y: 200, color: "#8b5cf6" },
  ];

  const connections = [
    { from: "input", to: "down1" },
    { from: "down1", to: "down2" },
    { from: "down2", to: "bottleneck" },
    { from: "bottleneck", to: "up1" },
    { from: "up1", to: "up2" },
    { from: "up2", to: "output" },
  ];

  const skipConnections = [
    { from: "down1", to: "up2", y: 250 },
    { from: "down2", to: "up1", y: 280 },
  ];

  return (
    <div className="relative w-full h-80">
      <svg className="w-full h-full" viewBox="0 0 750 350">
        {connections.map((conn, i) => {
          const fromBlock = blocks.find((b) => b.id === conn.from)!;
          const toBlock = blocks.find((b) => b.id === conn.to)!;
          return (
            <motion.line
              key={i}
              x1={fromBlock.x + 40}
              y1={fromBlock.y + 25}
              x2={toBlock.x}
              y2={toBlock.y + 25}
              stroke="#00f0ff"
              strokeWidth="2"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.6 }}
              transition={{ delay: i * 0.2, duration: 0.5 }}
            />
          );
        })}

        {skipConnections.map((skip, i) => {
          const fromBlock = blocks.find((b) => b.id === skip.from)!;
          const toBlock = blocks.find((b) => b.id === skip.to)!;
          return (
            <motion.path
              key={`skip-${i}`}
              d={`M ${fromBlock.x + 20} ${fromBlock.y + 50} 
                  Q ${fromBlock.x + 20} ${skip.y} ${(fromBlock.x + toBlock.x) / 2 + 20} ${skip.y}
                  Q ${toBlock.x + 20} ${skip.y} ${toBlock.x + 20} ${toBlock.y + 50}`}
              fill="none"
              stroke="#ff00aa"
              strokeWidth="2"
              strokeDasharray="5,5"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.8 }}
              transition={{ delay: 1.5 + i * 0.3, duration: 0.8 }}
            />
          );
        })}

        {blocks.map((block, i) => (
          <motion.g
            key={block.id}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.15, duration: 0.4, ease: "backOut" }}
          >
            <rect
              x={block.x}
              y={block.y}
              width={80}
              height={50}
              rx={8}
              fill="#0a0a0f"
              stroke={block.color}
              strokeWidth="2"
              filter="url(#glow)"
            />
            <text
              x={block.x + 40}
              y={block.y + 20}
              textAnchor="middle"
              fill="#e0e0e0"
              fontSize="10"
              fontFamily="JetBrains Mono"
            >
              {block.label.split("\n")[0]}
            </text>
            <text
              x={block.x + 40}
              y={block.y + 35}
              textAnchor="middle"
              fill="#6b7280"
              fontSize="8"
              fontFamily="JetBrains Mono"
            >
              {block.label.split("\n")[1]}
            </text>
          </motion.g>
        ))}

        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      </svg>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2 }}
        className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-6 text-xs"
      >
        <span className="flex items-center gap-2">
          <span className="w-3 h-3 rounded bg-neon-primary"></span>
          Encoder
        </span>
        <span className="flex items-center gap-2">
          <span className="w-3 h-3 rounded bg-neon-secondary"></span>
          Bottleneck
        </span>
        <span className="flex items-center gap-2">
          <span className="w-3 h-3 rounded bg-neon-accent"></span>
          Decoder
        </span>
        <span className="flex items-center gap-2">
          <span className="w-6 h-0.5 border-t-2 border-dashed border-neon-secondary"></span>
          Skip Connections
        </span>
      </motion.div>
    </div>
  );
}
