export type FeatureKey =
  | "rag"
  | "weather"
  | "geospatial"
  | "gradcam"
  | "memory"
  | "monitoring";

export type Role = "user" | "assistant";

export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  ts: number;
  attachments?: Attachments;
};

export type Attachments = {
  image?: File | null;
  audio?: File | null;
  location?: string;
};
