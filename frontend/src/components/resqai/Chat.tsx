"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useRef } from "react";
import type { ChatMessage } from "./types";
import { Button } from "@/components/ui/button";
import type { FeatureKey } from "./types";

function AttachmentRow(props: { msg: ChatMessage }) {
  const a = props.msg.attachments;
  const imageFile = a?.image ?? null;
  const imageUrl = useMemo(() => {
    if (!imageFile) return null;
    return URL.createObjectURL(imageFile);
  }, [imageFile]);

  useEffect(() => {
    if (!imageUrl) return;
    return () => URL.revokeObjectURL(imageUrl);
  }, [imageUrl]);

  if (!a) return null;

  const chips: Array<{ label: string }> = [];
  if (a.location?.trim()) chips.push({ label: a.location.trim() });
  if (a.audio) chips.push({ label: `audio · ${a.audio.name}` });

  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {a.image ? (
        <div className="overflow-hidden rounded-2xl border border-white/10 bg-white/5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageUrl ?? ""}
            alt="Uploaded"
            className="h-20 w-24 object-cover"
          />
        </div>
      ) : null}
      {chips.map((c) => (
        <span
          key={c.label}
          className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/75"
        >
          {c.label}
        </span>
      ))}
    </div>
  );
}

function Bubble(props: { msg: ChatMessage; showCursor: boolean }) {
  const isUser = props.msg.role === "user";
  const parts = useMemo(() => {
    if (isUser) return [props.msg.content];
    const raw = props.msg.content || "";
    return raw.split(/\n{2,}/g).filter((p) => p.trim().length > 0);
  }, [isUser, props.msg.content]);
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className={`mx-auto w-full max-w-3xl px-4`}
    >
      <div className="flex gap-4">
        <div
          className={`mt-1 size-7 shrink-0 rounded-full border ${
            isUser
              ? "border-white/15 bg-white/5"
              : "border-white/15 bg-white/10"
          }`}
          aria-hidden
        />
        <div className="flex-1">
          <div className="text-[15px] leading-7 text-white/92">
            {parts.map((p, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.18, delay: Math.min(0.06 * idx, 0.24) }}
                className="whitespace-pre-wrap"
              >
                {p}
                {!isUser && props.showCursor && idx === parts.length - 1 ? (
                  <span className="ml-0.5 inline-block h-[1.05em] w-[0.55ch] animate-pulse rounded-sm bg-white/70 align-[-0.15em]" />
                ) : null}
              </motion.div>
            ))}
            {isUser && props.showCursor ? (
              <span className="ml-0.5 inline-block h-[1.05em] w-[0.55ch] animate-pulse rounded-sm bg-white/70 align-[-0.15em]" />
            ) : null}
          </div>
          {isUser ? <AttachmentRow msg={props.msg} /> : null}
        </div>
      </div>
    </motion.div>
  );
}

function ActionPills(props: { onOpen: (k: FeatureKey) => void }) {
  const pills: Array<{ key: FeatureKey; label: string }> = [
    { key: "geospatial", label: "View Map" },
    { key: "rag", label: "View Sources" },
    { key: "gradcam", label: "View Flood Analysis" },
    { key: "memory", label: "View Memory" },
  ];
  return (
    <div className="mx-auto mt-3 w-full max-w-3xl px-4">
      <div className="flex flex-wrap gap-2 pl-11">
        {pills.map((p) => (
          <Button
            key={p.key}
            type="button"
            variant="outline"
            className="h-8 rounded-full border-white/12 bg-white/0 px-3 text-xs text-white/80 hover:bg-white/5 hover:text-white"
            onClick={() => props.onOpen(p.key)}
          >
            {p.label}
          </Button>
        ))}
      </div>
    </div>
  );
}

export function Chat(props: {
  messages: ChatMessage[];
  isStreaming: boolean;
  onOpenFeature: (k: FeatureKey) => void;
}) {
  const endRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [props.messages, props.isStreaming]);

  return (
    <div className="flex-1 overflow-auto">
      <div className="mx-auto w-full max-w-3xl px-4 pb-36 pt-10">
        <AnimatePresence>
          {props.messages.length === 0 ? (
            <motion.div
              key="hero"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.22 }}
              className="pt-16"
            >
              <div className="text-center">
                <div className="text-2xl font-semibold tracking-tight text-white">
                  ResQAI
                </div>
                <div className="mt-2 text-sm text-white/55">
                  Upload disaster information and ask what’s happening.
                </div>
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>

      <div className="space-y-7 pb-40">
        {props.messages.map((m, idx) => {
          const isLast = idx === props.messages.length - 1;
          const showCursor = props.isStreaming && isLast && m.role === "assistant";
          return <Bubble key={m.id} msg={m} showCursor={showCursor} />;
        })}

        {props.isStreaming ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mx-auto w-full max-w-3xl px-4"
          >
            <div className="flex gap-4">
              <div className="mt-1 size-7 shrink-0 rounded-full border border-white/15 bg-white/10" />
              <div className="flex items-center gap-2 text-sm text-white/55">
                <span className="inline-flex items-center gap-1">
                  <span className="inline-block size-1.5 animate-pulse rounded-full bg-white/60" />
                  <span className="inline-block size-1.5 animate-pulse rounded-full bg-white/45 [animation-delay:120ms]" />
                  <span className="inline-block size-1.5 animate-pulse rounded-full bg-white/30 [animation-delay:240ms]" />
                </span>
              </div>
            </div>
          </motion.div>
        ) : null}
      </div>

      {/* Show subtle action pills after the last assistant message completes */}
      {!props.isStreaming &&
      props.messages.length > 0 &&
      props.messages[props.messages.length - 1]?.role === "assistant" ? (
        <ActionPills onOpen={props.onOpenFeature} />
      ) : null}

      <div ref={endRef} />
    </div>
  );
}
