import { useEffect, useRef, useState, useCallback } from "react";
import type { ServerMessage } from "../types/messages";

/**
 * WebRTC-only browser stream.
 * Initiates WebRTC negotiation over the existing WebSocket as soon as connected.
 */
export function useBrowserStream(
  messages: ServerMessage[],
  send: (msg: any) => void,
  wsConnected: boolean
) {
  const [mode, setMode] = useState<"connecting" | "webrtc" | "error">("connecting");
  const [dimensions] = useState({ width: 1280, height: 800 });
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const offeredRef = useRef(false);
  const pendingStreamRef = useRef<MediaStream | null>(null);

  const startWebRTC = useCallback(async () => {
    if (offeredRef.current) return;
    offeredRef.current = true;

    try {
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      });
      pcRef.current = pc;

      pc.addTransceiver("video", { direction: "recvonly" });

      pc.ontrack = (event) => {
        const stream = event.streams[0];
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        } else {
          // video element not rendered yet — store and attach after mode switch
          pendingStreamRef.current = stream;
        }
        setMode("webrtc");
      };

      pc.onconnectionstatechange = () => {
        if (pc.connectionState === "failed" || pc.connectionState === "closed") {
          setMode("error");
          offeredRef.current = false;
        }
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      send({ type: "webrtc_offer", sdp: offer.sdp });
    } catch (e) {
      console.error("[WebRTC] setup failed", e);
      setMode("error");
    }
  }, [send]);

  // Handle webrtc_answer from server
  useEffect(() => {
    const latest = messages[messages.length - 1];
    if (!latest) return;
    if (latest.type === "webrtc_answer" && pcRef.current) {
      pcRef.current
        .setRemoteDescription(new RTCSessionDescription({ type: "answer", sdp: (latest as any).sdp }))
        .catch(() => setMode("error"));
    }
  }, [messages]);

  // Once mode switches to "webrtc" the <video> element renders — attach any pending stream
  useEffect(() => {
    if (mode === "webrtc" && pendingStreamRef.current && videoRef.current) {
      videoRef.current.srcObject = pendingStreamRef.current;
      pendingStreamRef.current = null;
    }
  }, [mode]);

  // Start WebRTC when WS connects
  useEffect(() => {
    if (wsConnected && mode === "connecting") {
      startWebRTC();
    }
  }, [wsConnected, mode, startWebRTC]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      pcRef.current?.close();
      pcRef.current = null;
    };
  }, []);

  return { mode, frameSrc: "", dimensions, videoRef };
}
