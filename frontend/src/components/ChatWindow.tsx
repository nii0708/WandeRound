"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Send, ChevronDown, ChevronRight, Loader2, Menu } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Message } from "@/types";

const MapView = dynamic(() => import("./MapView"), { ssr: false });

function ThinkingAccordion({ steps }: { steps: string[] }) {
  const [open, setOpen] = useState(false);
  if (!steps?.length) return null;
  return (
    <div>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-600 transition-colors"
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        {open ? "Hide" : "Show"} reasoning ({steps.length} steps)
      </button>
      {open && (
        <div className="mt-2 p-3 bg-zinc-50 rounded-lg border border-zinc-100 max-h-60 overflow-y-auto space-y-2">
          {steps.map((step, i) => (
            <div
              key={i}
              className="prose prose-xs prose-zinc max-w-none text-[11px] text-zinc-600
                prose-p:my-1 prose-p:leading-relaxed
                prose-headings:my-1 prose-headings:text-zinc-700
                prose-li:my-0.5
                prose-pre:my-1 prose-pre:bg-zinc-900 prose-pre:text-zinc-100 prose-pre:rounded-md prose-pre:p-2 prose-pre:text-[10px] prose-pre:overflow-x-auto
                prose-code:text-zinc-700 prose-code:bg-zinc-100 prose-code:px-1 prose-code:rounded prose-code:before:content-none prose-code:after:content-none"
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{step}</ReactMarkdown>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MessageBubble({
  msg,
  isLast,
  streaming,
}: {
  msg: Message;
  isLast: boolean;
  streaming: boolean;
}) {
  const isUser = msg.role === "user";
  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[78%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-zinc-900 text-white rounded-br-none"
            : "bg-white border border-zinc-100 text-zinc-800 rounded-bl-none shadow-sm"
        )}
      >
        {isLast && streaming && !msg.content ? (
          <div className="flex items-center gap-2 text-zinc-400">
            <Loader2 size={13} className="animate-spin" />
            <span className="text-xs">Thinking...</span>
          </div>
        ) : isUser ? (
          <p className="whitespace-pre-wrap">{msg.content}</p>
        ) : (
          <div className="prose prose-sm prose-zinc max-w-none
            prose-headings:font-semibold prose-headings:text-zinc-800
            prose-h3:text-sm prose-h3:mt-3 prose-h3:mb-1
            prose-p:text-zinc-700 prose-p:my-1.5
            prose-li:text-zinc-700 prose-li:my-0.5
            prose-ul:my-1.5 prose-ol:my-1.5
            prose-strong:text-zinc-800 prose-strong:font-semibold
            prose-code:text-zinc-700 prose-code:bg-zinc-100 prose-code:px-1 prose-code:rounded
            prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {msg.content}
            </ReactMarkdown>
          </div>
        )}

        {!isUser &&
          msg.thinking_steps &&
          msg.thinking_steps.length > 0 && (
            <div className="mt-2">
              <ThinkingAccordion steps={msg.thinking_steps} />
            </div>
          )}

        {!isUser && msg.geojson && (
          <MapView geojson={msg.geojson} labels={msg.map_labels} />
        )}
      </div>
    </div>
  );
}

interface Props {
  messages: Message[];
  isStreaming: boolean;
  onSendMessage: (input: string) => void;
  hasThread: boolean;
  onCreateThread: () => void;
  onOpenSidebar?: () => void;
}

export function ChatWindow({
  messages,
  isStreaming,
  onSendMessage,
  hasThread,
  onCreateThread,
  onOpenSidebar,
}: Props) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;
    onSendMessage(trimmed);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const autoResize = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 128)}px`;
  };

  if (!hasThread) {
    return (
      <div className="flex-1 flex flex-col bg-zinc-50 min-w-0">
        <div className="md:hidden flex items-center px-3 py-2 border-b border-zinc-100 bg-white">
          <button
            onClick={onOpenSidebar}
            className="p-2 rounded-lg hover:bg-zinc-100 text-zinc-700"
            aria-label="Open sidebar"
          >
            <Menu size={18} />
          </button>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center gap-6 px-4">
          <Image
            src="/wanderound-icon.png"
            alt="WandeRound"
            width={64}
            height={64}
            priority
          />
          <div className="text-center space-y-2">
            <h1 className="text-3xl font-semibold tracking-tight text-zinc-900">
              WandeRound
            </h1>
            <p className="text-sm text-zinc-500 max-w-xs">
              Plan your next trip with AI-powered geospatial intelligence
            </p>
          </div>
          <button
            onClick={onCreateThread}
            className="px-5 py-2.5 bg-zinc-900 text-white text-sm rounded-xl hover:bg-zinc-700 transition-colors font-medium"
          >
            Start planning
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-zinc-50 min-w-0 overflow-hidden">
      {/* Mobile header with hamburger */}
      <div className="md:hidden flex items-center gap-2 px-3 py-2 border-b border-zinc-100 bg-white">
        <button
          onClick={onOpenSidebar}
          className="p-2 rounded-lg hover:bg-zinc-100 text-zinc-700"
          aria-label="Open sidebar"
        >
          <Menu size={18} />
        </button>
        <span className="text-sm font-medium text-zinc-700 truncate">
          WandeRound
        </span>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-4"
      >
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-zinc-400">
              Describe a trip and I'll help you plan it
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            msg={msg}
            isLast={i === messages.length - 1}
            streaming={isStreaming}
          />
        ))}
      </div>

      {/* Input */}
      <div className="px-4 md:px-8 py-4 bg-white border-t border-zinc-100">
        <div className="flex items-end gap-2 md:gap-3 max-w-3xl mx-auto">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={autoResize}
            onKeyDown={handleKeyDown}
            placeholder="Plan a 2-day trip to Yogyakarta..."
            rows={1}
            disabled={isStreaming}
            className="flex-1 resize-none rounded-xl border border-zinc-200 px-4 py-3 text-sm text-zinc-900 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-900 focus:border-transparent bg-white disabled:opacity-50 overflow-hidden"
            style={{ minHeight: "48px", maxHeight: "128px" }}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
            className="w-10 h-10 rounded-xl bg-zinc-900 text-white flex items-center justify-center hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex-shrink-0"
          >
            {isStreaming ? (
              <Loader2 size={15} className="animate-spin" />
            ) : (
              <Send size={15} />
            )}
          </button>
        </div>
        <p className="text-center text-[11px] text-zinc-400 mt-2.5">
          Powered by DeepSeek · OpenStreetMap
        </p>
      </div>
    </div>
  );
}
