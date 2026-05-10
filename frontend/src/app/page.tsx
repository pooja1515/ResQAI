"use client";

import { useMemo, useState } from "react";
import { Sidebar } from "@/components/resqai/Sidebar";
import { Chat } from "@/components/resqai/Chat";
import { Composer } from "@/components/resqai/Composer";
import { FeatureOverlay } from "@/components/resqai/FeatureOverlay";
import type { FeatureKey } from "@/components/resqai/types";
import { useStreamingOrchestration } from "@/components/resqai/useStreamingOrchestration";

export default function Home() {
  const [overlay, setOverlay] = useState<FeatureKey | null>(null);
  const orch = useStreamingOrchestration();

  const history = useMemo(
    () => ["Incident #0412 · Mumbai", "Incident #0409 · Chennai", "Incident #0401 · Manila"],
    []
  );

  return (
    <div className="flex h-dvh w-full bg-black text-white">
      <Sidebar
        history={history}
        onNew={() => orch.reset()}
        onOpenFeature={(k) => setOverlay(k)}
      />

      <main className="relative flex h-dvh flex-1 flex-col">
        <Chat
          messages={orch.messages}
          isStreaming={orch.isStreaming}
          onOpenFeature={(k) => setOverlay(k)}
        />

        <div className="pointer-events-none absolute bottom-3 left-0 right-0">
          <Composer
            disabled={orch.isStreaming}
            onSend={async (text, attachments) => {
              await orch.send(text, attachments);
            }}
          />
        </div>
      </main>

      <FeatureOverlay
        open={overlay !== null}
        feature={overlay}
        onClose={() => setOverlay(null)}
      />
    </div>
  );
}
