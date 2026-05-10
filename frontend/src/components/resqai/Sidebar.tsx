"use client";

import { Plus, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { FeatureKey } from "./types";

const FEATURES: Array<{ key: FeatureKey; label: string }> = [
  { key: "rag", label: "RAG" },
  { key: "weather", label: "Weather" },
  { key: "geospatial", label: "Geospatial" },
  { key: "gradcam", label: "Grad-CAM" },
  { key: "memory", label: "Memory" },
  { key: "monitoring", label: "Monitoring" },
];

export function Sidebar(props: {
  onNew: () => void;
  onOpenFeature: (key: FeatureKey) => void;
  history: string[];
}) {
  return (
    <aside className="h-dvh w-[280px] shrink-0 border-r border-white/10 bg-black">
      <div className="flex h-full flex-col p-3">
        <div className="flex items-center gap-2 px-2 py-2">
          <div className="grid size-8 place-items-center rounded-lg border border-white/10 bg-white/5">
            <Sparkles className="size-4 text-white/90" />
          </div>
          <div className="flex flex-col leading-tight">
            <div className="text-sm font-semibold tracking-tight">ResQAI</div>
            <div className="text-xs text-white/55">Disaster intelligence</div>
          </div>
        </div>

        <div className="mt-2 px-1">
          <Button
            onClick={props.onNew}
            className="w-full justify-start gap-2 rounded-xl bg-white text-black hover:bg-white/90"
          >
            <Plus className="size-4" />
            New Analysis
          </Button>
        </div>

        <div className="mt-4 px-2 text-xs font-medium text-white/55">
          History
        </div>
        <div className="mt-2 flex-1 overflow-auto px-1">
          <ul className="space-y-1">
            {props.history.map((h) => (
              <li key={h}>
                <button
                  className="w-full rounded-xl px-3 py-2 text-left text-sm text-white/80 hover:bg-white/4 hover:text-white/90 transition-colors"
                  type="button"
                >
                  {h}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="mt-3 border-t border-white/10 pt-3">
          <div className="mt-1 flex flex-wrap gap-2 px-1">
            {FEATURES.map((f) => (
              <button
                key={f.key}
                type="button"
                onClick={() => props.onOpenFeature(f.key)}
                className="rounded-full border border-white/10 bg-white/0 px-3 py-1.5 text-sm text-white/75 hover:bg-white/5 hover:text-white/90 transition-colors"
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
}
