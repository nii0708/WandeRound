"use client";

import { useState, useEffect, useCallback } from "react";
import { ChatSidebar } from "@/components/ChatSidebar";
import { ChatWindow } from "@/components/ChatWindow";
import type { Thread, Message, SSEEvent, GeoJSONData } from "@/types";

export default function Home() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const fetchThreads = useCallback(async () => {
    const res = await fetch("/api/threads");
    setThreads(await res.json());
  }, []);

  const fetchMessages = useCallback(async (threadId: string) => {
    const res = await fetch(`/api/threads/${threadId}/messages`);
    const data: Message[] = await res.json();

    // Enrich historical messages that have a geopandas_link but no geojson
    const enriched = await Promise.all(
      data.map(async (msg) => {
        if (msg.geopandas_link && !msg.geojson) {
          try {
            const r = await fetch(
              `/api/map?file=${encodeURIComponent(msg.geopandas_link)}`
            );
            if (r.ok) return { ...msg, geojson: (await r.json()) as GeoJSONData };
          } catch {
            // file might be missing, just skip
          }
        }
        return msg;
      })
    );
    setMessages(enriched);
  }, []);

  useEffect(() => {
    fetchThreads();
  }, [fetchThreads]);

  useEffect(() => {
    if (currentThreadId) fetchMessages(currentThreadId);
    else setMessages([]);
  }, [currentThreadId, fetchMessages]);

  const createThread = async () => {
    const res = await fetch("/api/threads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await res.json();
    await fetchThreads();
    setCurrentThreadId(data.id);
  };

  const deleteThread = async (threadId: string) => {
    await fetch(`/api/threads/${threadId}`, { method: "DELETE" });
    await fetchThreads();
    if (currentThreadId === threadId) setCurrentThreadId(null);
  };

  const renameThread = async (threadId: string, title: string) => {
    await fetch(`/api/threads/${threadId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    await fetchThreads();
  };

  const sendMessage = async (input: string) => {
    if (!currentThreadId || isStreaming) return;

    // Optimistically add user message
    const userMsg: Message = {
      role: "user",
      content: input,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsStreaming(true);

    // Save user message
    await fetch(`/api/threads/${currentThreadId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role: "user", content: input }),
    });

    // Auto-title thread from first message
    if (messages.length === 0) {
      await renameThread(currentThreadId, input.slice(0, 45));
    }

    // Add empty assistant message placeholder
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", timestamp: new Date().toISOString(), thinking_steps: [] },
    ]);

    let finalContent = "";
    let finalThinking: string[] = [];
    let geodataFile: string | null = null;
    let mapLabels: import("@/types").MapLabels | null = null;

    try {
      const backendUrl =
        process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const response = await fetch(`${backendUrl}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: currentThreadId, message: input }),
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event: SSEEvent = JSON.parse(line.slice(6));

            if (event.type === "thinking") {
              const added = [...(event.steps ?? []), ...(event.overpass ?? [])];
              finalThinking = [...finalThinking, ...added];
              setMessages((prev) => {
                const msgs = [...prev];
                msgs[msgs.length - 1] = {
                  ...msgs[msgs.length - 1],
                  thinking_steps: finalThinking,
                };
                return msgs;
              });
            }

            if (event.type === "overpass") {
              finalThinking = [...finalThinking, ...(event.queries ?? [])];
              setMessages((prev) => {
                const msgs = [...prev];
                msgs[msgs.length - 1] = {
                  ...msgs[msgs.length - 1],
                  thinking_steps: finalThinking,
                };
                return msgs;
              });
            }

            if (event.type === "code") {
              finalThinking = [...finalThinking, event.content ?? ""];
              setMessages((prev) => {
                const msgs = [...prev];
                msgs[msgs.length - 1] = {
                  ...msgs[msgs.length - 1],
                  thinking_steps: finalThinking,
                };
                return msgs;
              });
            }

            if (event.type === "map") {
              geodataFile = event.file ?? null;
              mapLabels = event.labels ?? null;
              setMessages((prev) => {
                const msgs = [...prev];
                msgs[msgs.length - 1] = {
                  ...msgs[msgs.length - 1],
                  geojson: event.geojson,
                  map_labels: event.labels ?? undefined,
                };
                return msgs;
              });
            }

            if (event.type === "delta") {
              finalContent += event.content ?? "";
              setMessages((prev) => {
                const msgs = [...prev];
                msgs[msgs.length - 1] = {
                  ...msgs[msgs.length - 1],
                  content: finalContent,
                };
                return msgs;
              });
            }

            if (event.type === "response") {
              if (event.content) {
                finalContent = event.content;
                setMessages((prev) => {
                  const msgs = [...prev];
                  msgs[msgs.length - 1] = {
                    ...msgs[msgs.length - 1],
                    content: finalContent,
                  };
                  return msgs;
                });
              }
            }

            if (event.type === "done") {
              finalContent = event.content || finalContent;
              if (event.labels) mapLabels = event.labels;
              await fetch(`/api/threads/${currentThreadId}/messages`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  role: "assistant",
                  content: finalContent,
                  thinking_steps: finalThinking,
                  geopandas_link: geodataFile,
                  map_labels: mapLabels,
                }),
              });
              await fetchThreads();
            }

            if (event.type === "error") {
              setMessages((prev) => {
                const msgs = [...prev];
                msgs[msgs.length - 1] = {
                  ...msgs[msgs.length - 1],
                  content: `Error: ${event.content}`,
                };
                return msgs;
              });
            }
          } catch {
            // ignore parse errors
          }
        }
      }
    } catch {
      setMessages((prev) => {
        const msgs = [...prev];
        msgs[msgs.length - 1] = {
          ...msgs[msgs.length - 1],
          content: "Connection error. Please try again.",
        };
        return msgs;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleSelectThread = (id: string) => {
    setCurrentThreadId(id);
    setSidebarOpen(false);
  };

  const handleCreateThread = async () => {
    await createThread();
    setSidebarOpen(false);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-white relative">
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 bg-black/50 z-[998] md:hidden"
          aria-hidden
        />
      )}
      <ChatSidebar
        threads={threads}
        currentThreadId={currentThreadId}
        onSelectThread={handleSelectThread}
        onCreateThread={handleCreateThread}
        onDeleteThread={deleteThread}
        onRenameThread={renameThread}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <ChatWindow
        messages={messages}
        isStreaming={isStreaming}
        onSendMessage={sendMessage}
        hasThread={!!currentThreadId}
        onCreateThread={createThread}
        onOpenSidebar={() => setSidebarOpen(true)}
      />
    </div>
  );
}
