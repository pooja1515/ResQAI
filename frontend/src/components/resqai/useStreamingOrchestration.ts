"use client";

import { useCallback, useMemo, useState } from "react";
import type { Attachments, ChatMessage } from "./types";

type StreamEvent =
  | { type: "status"; text: string }
  | { type: "token"; text: string }
  | { type: "error"; message: string }
  | { type: "done" };

function now() {
  return Date.now();
}

function id() {
  return globalThis.crypto?.randomUUID?.() ?? String(Date.now() + Math.random());
}

function apiBase() {
  return (
    process.env.NEXT_PUBLIC_API_BASE?.trim() ||
    "http://127.0.0.1:8000"
  );
}

async function* streamAnalyze(
  prompt: string,
  attachments: Attachments
): AsyncGenerator<StreamEvent> {
  const controller = new AbortController();
  const timeoutMs = 90_000;
  const t = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${apiBase()}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        message: prompt,
        location: attachments.location || undefined,
      }),
    });

    if (!res.ok) {
      yield {
        type: "error",
        message:
          "I couldn’t reach the ResQAI engine right now. Please try again in a moment.",
      };
      return;
    }

    const body = res.body;
    if (!body) {
      yield {
        type: "error",
        message:
          "I couldn’t read the ResQAI response stream. Please try again.",
      };
      return;
    }

    const reader = body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const lines = frame
          .split("\n")
          .map((l) => l.trimEnd())
          .filter(Boolean);

        let eventName = "message";
        let dataLine = "";
        for (const line of lines) {
          if (line.startsWith("event:")) eventName = line.slice(6).trim();
          if (line.startsWith("data:")) dataLine += line.slice(5).trim();
        }

        if (!dataLine) continue;
        let payload: unknown = null;
        try {
          payload = JSON.parse(dataLine);
        } catch {
          continue;
        }

        const p = payload as Record<string, unknown> | null;

        if (eventName === "status" && typeof p?.text === "string") {
          yield { type: "status", text: p.text };
        } else if (
          eventName === "token" &&
          typeof p?.text === "string"
        ) {
          yield { type: "token", text: p.text };
        } else if (
          eventName === "error" &&
          typeof p?.message === "string"
        ) {
          yield { type: "error", message: p.message };
          return;
        } else if (eventName === "done") {
          yield { type: "done" };
          return;
        }
      }
    }
  } catch {
    yield {
      type: "error",
      message:
        "ResQAI is unavailable right now (connection or timeout). Please try again.",
    };
  } finally {
    clearTimeout(t);
  }
}

export function useStreamingOrchestration() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const send = useCallback(async (text: string, attachments: Attachments) => {
    const content = text.trim().length ? text.trim() : "(inputs provided)";
    setMessages((prev) => [
      ...prev,
      { id: id(), role: "user", content, ts: now(), attachments },
    ]);

    setIsStreaming(true);

    // One assistant message that streams (status lines then final)
    const assistantId = id();
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "", ts: now() },
    ]);

    const stream = streamAnalyze(content, attachments);
    for await (const ev of stream) {
      if (ev.type === "status") {
        // Add a new paragraph-style status line.
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content:
                    (m.content ? m.content + "\n\n" : "") +
                    ev.text.replace(/\s+$/g, ""),
                }
              : m
          )
        );
      } else if (ev.type === "token") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: m.content + ev.text } : m
          )
        );
      } else if (ev.type === "error") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content:
                    (m.content ? m.content + "\n\n" : "") + ev.message,
                }
              : m
          )
        );
        break;
      } else if (ev.type === "done") {
        break;
      }
    }

    setIsStreaming(false);
  }, []);

  const api = useMemo(
    () => ({
      messages,
      isStreaming,
      send,
      reset: () => setMessages([]),
    }),
    [messages, isStreaming, send]
  );

  return api;
}
