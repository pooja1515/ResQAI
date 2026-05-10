"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import type { FeatureKey } from "./types";

const TITLES: Record<FeatureKey, string> = {
  rag: "RAG Knowledge",
  weather: "Weather Intelligence",
  geospatial: "Geospatial",
  gradcam: "Grad‑CAM",
  memory: "Memory",
  monitoring: "Monitoring",
};

export function FeatureOverlay(props: {
  open: boolean;
  feature: FeatureKey | null;
  onClose: () => void;
}) {
  const title = props.feature ? TITLES[props.feature] : "Feature";
  const body =
    props.feature === "rag"
      ? {
          heading: "Grounded sources",
          text: "Show retrieved chunks, citations, and grounded response here.",
        }
      : props.feature === "weather"
      ? {
          heading: "Weather intelligence",
          text: "Show live weather risk assessment and escalation notes here.",
        }
      : props.feature === "geospatial"
      ? {
          heading: "Operational map",
          text: "Embed the interactive Folium map (HTML) or a tiles-based map here.",
        }
      : props.feature === "gradcam"
      ? {
          heading: "Vision explainability",
          text: "Show original image, heatmap, and overlay artifacts here.",
        }
      : props.feature === "memory"
      ? {
          heading: "Temporal memory",
          text: "Show minimal timeline trend + summaries here.",
        }
      : {
          heading: "Agent monitoring",
          text: "Show lightweight orchestration feed and agent statuses here.",
        };

  return (
    <AnimatePresence>
      {props.open ? (
        <motion.div
          className="fixed inset-0 z-50"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <div
            className="absolute inset-0 bg-black/70 backdrop-blur-sm"
            onClick={props.onClose}
          />

          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.98 }}
            transition={{ duration: 0.18 }}
            className="absolute left-1/2 top-1/2 w-[92vw] max-w-3xl -translate-x-1/2 -translate-y-1/2 rounded-3xl border border-white/12 bg-black p-5 shadow-[0_40px_120px_rgba(0,0,0,0.85)]"
          >
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-white/92">{title}</div>
              <button
                type="button"
                onClick={props.onClose}
                className="rounded-xl border border-white/10 bg-white/5 p-2 text-white/70 hover:bg-white/8 hover:text-white"
              >
                <X className="size-4" />
              </button>
            </div>

            <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/70">
              <div className="text-white/90">{body.heading}</div>
              <div className="mt-1 text-white/55">{body.text}</div>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
