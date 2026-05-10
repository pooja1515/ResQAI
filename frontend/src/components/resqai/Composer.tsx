"use client";

import { useRef, useState } from "react";
import { Mic, Paperclip, Send, MapPin, Image as ImageIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Attachments } from "./types";

export function Composer(props: {
  disabled?: boolean;
  onSend: (text: string, attachments: Attachments) => void;
}) {
  const [text, setText] = useState("");
  const [location, setLocation] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [audio, setAudio] = useState<File | null>(null);

  const imageRef = useRef<HTMLInputElement | null>(null);
  const audioRef = useRef<HTMLInputElement | null>(null);

  const canSend = text.trim().length > 0 || image || audio || location.trim();

  return (
    <div className="pointer-events-auto mx-auto w-full max-w-3xl px-4 pb-6">
      <div className="group rounded-[28px] border border-white/10 bg-white/[0.04] p-2.5 shadow-[0_0_0_1px_rgba(255,255,255,0.05),0_18px_40px_rgba(0,0,0,0.55)] transition-shadow focus-within:shadow-[0_0_0_1px_rgba(255,255,255,0.10),0_0_18px_rgba(255,255,255,0.06),0_22px_50px_rgba(0,0,0,0.65)]">
        <div className="flex items-center gap-2 px-2 pb-1.5 text-[11px] text-white/55">
          <span className="inline-flex items-center gap-1">
            <MapPin className="size-3.5" />
            <input
              className="w-[170px] bg-transparent outline-none placeholder:text-white/30"
              placeholder="Location (optional)"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            />
          </span>
          <div className="ml-auto flex items-center gap-2">
            {image ? (
              <span className="rounded-full border border-white/10 bg-white/5 px-2 py-1">
                image: {image.name}
              </span>
            ) : null}
            {audio ? (
              <span className="rounded-full border border-white/10 bg-white/5 px-2 py-1">
                audio: {audio.name}
              </span>
            ) : null}
          </div>
        </div>

        <textarea
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Describe what’s happening…"
          className="min-h-[44px] w-full resize-none bg-transparent px-3 py-2 text-[15px] leading-6 text-white/90 outline-none placeholder:text-white/35"
        />

        <div className="mt-1.5 flex items-center justify-between gap-2 px-2">
          <div className="flex items-center gap-1">
            <input
              ref={imageRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => setImage(e.target.files?.[0] ?? null)}
            />
            <input
              ref={audioRef}
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => setAudio(e.target.files?.[0] ?? null)}
            />

            <Button
              type="button"
              variant="ghost"
              className="h-8 w-9 rounded-full bg-transparent p-0 text-white/75 hover:bg-white/7 hover:text-white"
              onClick={() => imageRef.current?.click()}
              aria-label="Attach image"
              title="Attach image"
            >
              <ImageIcon className="size-4" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              className="h-8 w-9 rounded-full bg-transparent p-0 text-white/75 hover:bg-white/7 hover:text-white"
              onClick={() => audioRef.current?.click()}
              aria-label="Attach audio"
              title="Attach audio"
            >
              <Mic className="size-4" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              className="h-8 w-9 rounded-full bg-transparent p-0 text-white/75 hover:bg-white/7 hover:text-white"
              onClick={() => {
                setImage(null);
                setAudio(null);
                setLocation("");
              }}
              aria-label="Clear attachments"
              title="Clear"
            >
              <Paperclip className="size-4" />
            </Button>
          </div>

          <Button
            type="button"
            disabled={props.disabled || !canSend}
            onClick={() => {
              props.onSend(text, { image, audio, location });
              setText("");
            }}
            className="h-8 rounded-full bg-white px-4 text-sm text-black hover:bg-white/90"
          >
            <Send className="mr-2 size-4" />
            Send
          </Button>
        </div>
      </div>
    </div>
  );
}
