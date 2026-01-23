import React from "react";
import NeonCard from "../../components/NeonCard";
import FormulaBlock from "../../components/FormulaBlock";
import GlitchText from "../../components/GlitchText";

export default function SlideForward() {
  return (
    <div className="flex flex-col h-full">
      <div className="mb-10">
        <GlitchText text="Forward Process" className="text-7xl font-bold text-neon-primary mb-2" />
        <p className="text-2xl text-neon-muted">Processus de bruitage progressif q(xₜ|x₀)</p>
      </div>

      <div className="grid grid-cols-2 gap-8 flex-1 min-h-0">
        <div className="space-y-8 flex flex-col h-full">
          <NeonCard title="Distribution conditionnelle" glow="cyan" className="flex-1">
            <div className="flex items-center justify-center h-full pt-4">
              <FormulaBlock 
                formula="q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I})" 
              />
            </div>
          </NeonCard>

          <NeonCard title="Reparameterization Trick" glow="magenta" className="flex-1">
            <div className="flex items-center justify-center h-full pt-4">
              <FormulaBlock 
                formula="x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon" 
              />
            </div>
          </NeonCard>
        </div>

        <div className="space-y-8 flex flex-col h-full">
          <NeonCard title="Schedule linéaire des β" glow="violet" className="flex-1">
            <div className="space-y-4 font-mono text-lg mt-6 px-4">
              <div className="flex justify-between border-b border-neon-accent/20 pb-2">
                <span className="text-neon-accent font-bold">β₁ =</span>
                <span className="text-white">0.0001</span>
              </div>
              <div className="flex justify-between border-b border-neon-accent/20 pb-2">
                <span className="text-neon-accent font-bold">βₜ =</span>
                <span className="text-white">0.02</span>
              </div>
              <div className="flex justify-between border-b border-neon-accent/20 pb-2">
                <span className="text-neon-accent font-bold">T =</span>
                <span className="text-white">1000 steps</span>
              </div>
            </div>
          </NeonCard>

          <NeonCard title="Définitions" glow="cyan" className="flex-1">
            <div className="space-y-4 flex flex-col justify-center h-full pt-4 pb-2">
              <FormulaBlock formula="\alpha_t = 1 - \beta_t" />
              <FormulaBlock formula="\bar{\alpha}_t = \prod_{s=1}^t \alpha_s" />
            </div>
          </NeonCard>
        </div>
      </div>
    </div>
  );
}
