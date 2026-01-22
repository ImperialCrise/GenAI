"use client";

import { useParams } from "next/navigation";
import SlideContainer from "@/components/SlideContainer";
import Slide01Intro from "@/lib/slides/01-intro";
import Slide02Motivation from "@/lib/slides/02-motivation";
import Slide03Overview from "@/lib/slides/03-overview";
import Slide04Forward from "@/lib/slides/04-forward";
import Slide05UNet from "@/lib/slides/05-unet";
import Slide06Reverse from "@/lib/slides/06-reverse";
import Slide07Training from "@/lib/slides/07-training";
import Slide08VizForward from "@/lib/slides/08-viz-forward";
import Slide09VizReverse from "@/lib/slides/09-viz-reverse";
import Slide10Results from "@/lib/slides/10-results";
import Slide11Comparison from "@/lib/slides/11-comparison";
import Slide12Merci from "@/lib/slides/12-merci";

const SLIDE_COMPONENTS: Record<number, React.ComponentType> = {
  1: Slide01Intro,
  2: Slide02Motivation,
  3: Slide03Overview,
  4: Slide04Forward,
  5: Slide05UNet,
  6: Slide06Reverse,
  7: Slide07Training,
  8: Slide08VizForward,
  9: Slide09VizReverse,
  10: Slide10Results,
  11: Slide11Comparison,
  12: Slide12Merci,
};

export default function SlidePage() {
  const params = useParams();
  const slideId = parseInt(params.id as string, 10);

  const SlideComponent = SLIDE_COMPONENTS[slideId];

  if (!SlideComponent) {
    return (
      <SlideContainer slideNumber={1}>
        <div className="flex-1 flex items-center justify-center">
          <p className="text-neon-secondary text-2xl">Slide non trouvée</p>
        </div>
      </SlideContainer>
    );
  }

  return (
    <SlideContainer slideNumber={slideId}>
      <SlideComponent />
    </SlideContainer>
  );
}
