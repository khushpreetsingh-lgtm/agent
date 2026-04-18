import React, { useCallback } from "react";
import type { ServerMessage } from "../types/messages";
import { useBrowserStream } from "../hooks/useBrowserStream";

interface Props {
  messages: ServerMessage[];
  send: (msg: any) => void;
  viewportWidth: number;
  viewportHeight: number;
  wsConnected: boolean;
}

export function BrowserPanel({ messages, send, viewportWidth, viewportHeight, wsConnected }: Props) {
  const { mode, dimensions, videoRef } = useBrowserStream(messages, send, wsConnected);

  const scaleCoords = useCallback(
    (clientX: number, clientY: number) => {
      const el = videoRef.current;
      if (!el) return { x: 0, y: 0 };
      const rect = el.getBoundingClientRect();
      const x = ((clientX - rect.left) / rect.width) * viewportWidth;
      const y = ((clientY - rect.top) / rect.height) * viewportHeight;
      return { x: Math.round(x), y: Math.round(y) };
    },
    [viewportWidth, viewportHeight, mode, videoRef]
  );

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      const { x, y } = scaleCoords(e.clientX, e.clientY);
      send({ type: "browser_input", action: "click", x, y });
    },
    [scaleCoords, send]
  );

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      const { x, y } = scaleCoords(e.clientX, e.clientY);
      send({ type: "browser_input", action: "dblclick", x, y });
    },
    [scaleCoords, send]
  );

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      const { x, y } = scaleCoords(e.clientX, e.clientY);
      send({ type: "browser_input", action: "scroll", x, y, delta: e.deltaY });
    },
    [scaleCoords, send]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      e.preventDefault();
      if (e.key.length === 1) {
        send({ type: "browser_input", action: "type", text: e.key });
      } else {
        send({ type: "browser_input", action: "key", key: e.key });
      }
    },
    [send]
  );

  const streamLabel = mode === "webrtc" ? "WebRTC" : mode === "error" ? "Error" : "...";

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-lg overflow-hidden">
      <div className="px-3 py-2 bg-gray-800 text-sm text-gray-400 flex items-center justify-between">
        <span>Live Browser</span>
        <span className="text-xs">
          {dimensions.width}x{dimensions.height} | {streamLabel}
        </span>
      </div>
      <div
        className="flex-1 relative bg-black flex items-center justify-center overflow-hidden"
        tabIndex={0}
        onKeyDown={handleKeyDown}
      >
        {mode === "webrtc" ? (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="max-w-full max-h-full object-contain cursor-crosshair"
            onClick={handleClick}
            onDoubleClick={handleDoubleClick}
            onWheel={handleWheel}
          />
        ) : mode === "error" ? (
          <div className="text-red-500 text-sm">WebRTC connection failed</div>
        ) : (
          <div className="text-gray-600 text-sm">Connecting via WebRTC...</div>
        )}
      </div>
    </div>
  );
}
