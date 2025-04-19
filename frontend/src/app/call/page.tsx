"use client"

import { useEffect, useRef, useState } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AIAnimation } from "@/components/AIAnimation"

const URL_WEB_SOCKET = "ws://localhost:8090/ws"

let localStream: MediaStream
let localPeerConnection: RTCPeerConnection
let mediaRecorder: MediaRecorder | null = null
const recordedChunks: Blob[] = []

const pcConstraints: RTCConfiguration = {
    optional: [{ DtlsSrtpKeyAgreement: true }],
}

export default function CallPage() {
    const ws = useRef<WebSocket | null>(null)
    const searchParams = useSearchParams()
    const router = useRouter()
    const [isMuted, setIsMuted] = useState(false)

    const handleMute = () => {
        if (localStream) {
            const audioTracks = localStream.getAudioTracks()
            audioTracks.forEach((track) => {
                track.enabled = !track.enabled
            })
            setIsMuted(!isMuted)
        }
    }

    const handleEndCall = () => {
        // Stop all media tracks
        if (localStream) {
            localStream.getTracks().forEach((track) => track.stop())
        }

        // Close WebSocket connection
        if (ws.current) {
            ws.current.close()
        }

        // Close peer connection
        if (localPeerConnection) {
            localPeerConnection.close()
        }

        // Stop media recorder if active
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop()
        }

        // Navigate back to home page
        router.push("/")
    }

    useEffect(() => {
        const wsClient = new WebSocket(URL_WEB_SOCKET)
        wsClient.onopen = () => {
            console.log("ws opened")
            ws.current = wsClient
            setupDevice()
        }
        wsClient.onclose = () => console.log("Ws closed")
        wsClient.onmessage = (message: MessageEvent) => {
            const parsedMessage = JSON.parse(message.data)
            const { type, body } = parsedMessage
            switch (type) {
                case "joined":
                    console.log("Users in this channel", body)
                    break
                case "offer_sdp_received":
                    const offer = body
                    onAnswer(offer)
                    break
                case "answer_sdp_received":
                    gotRemoteDescription(body)
                    break
                case "ice_candidate_received":
                    break
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const gotRemoteDescription = (answer: RTCSessionDescriptionInit) => {
        localPeerConnection.setRemoteDescription(new RTCSessionDescription(answer))
        localPeerConnection.onaddstream = gotRemoteStream
    }

    const onAnswer = async (offer: RTCSessionDescriptionInit) => {
        console.log("onAnswer invoked")
        localPeerConnection = new RTCPeerConnection(pcConstraints)
        localPeerConnection.onicecandidate = gotLocalIceCandidateAnswer
        localPeerConnection.onaddstream = gotRemoteStream
        localPeerConnection.addStream(localStream)

        try {
            await localPeerConnection.setRemoteDescription(new RTCSessionDescription(offer))
            const answer = await localPeerConnection.createAnswer()
            await localPeerConnection.setLocalDescription(answer)
        } catch (error) {
            console.error("Error in onAnswer:", error)
        }
    }

    const gotAnswerDescription = async (answer: RTCSessionDescriptionInit) => {
        try {
            await localPeerConnection.setLocalDescription(new RTCSessionDescription(answer))
        } catch (error) {
            console.error("Error in gotAnswerDescription:", error)
        }
    }

    const gotLocalIceCandidateAnswer = (event: RTCPeerConnectionIceEvent) => {
        if (!event.candidate) {
            const answer = localPeerConnection.localDescription
            sendWsMessage("send_answer", {
                channelName: searchParams.get("channelName"),
                userName: searchParams.get("userName"),
                sdp: answer,
            })
        }
    }

    const sendWsMessage = (type: string, body: any) => {
        ws.current?.send(JSON.stringify({ type, body }))
    }

    const setupPeerConnection = async () => {
        console.log("Setting up peer connection")
        localPeerConnection = new RTCPeerConnection(pcConstraints)
        localPeerConnection.onicecandidate = gotLocalIceCandidateOffer
        localPeerConnection.onaddstream = gotRemoteStream
        localPeerConnection.addStream(localStream)

        try {
            const offer = await localPeerConnection.createOffer()
            await localPeerConnection.setLocalDescription(offer)
        } catch (error) {
            console.error("Error in setupPeerConnection:", error)
        }
    }

    const gotLocalDescription = async (offer: RTCSessionDescriptionInit) => {
        try {
            await localPeerConnection.setLocalDescription(new RTCSessionDescription(offer))
        } catch (error) {
            console.error("Error in gotLocalDescription:", error)
        }
    }

    const gotRemoteStream = (event: MediaStreamEvent) => {
        const remotePlayer = document.getElementById("peerPlayer") as HTMLVideoElement | null
        if (remotePlayer) {
            remotePlayer.srcObject = event.stream

            // Start recording the peer's stream
            mediaRecorder = new MediaRecorder(event.stream, {
                mimeType: "video/webm;codecs=vp8,opus",
            })

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    recordedChunks.push(e.data)
                    // Send the chunk to backend
                    sendWsMessage("video_chunk", {
                        channelName: searchParams.get("channelName"),
                        userName: searchParams.get("userName"),
                        chunk: e.data,
                    })
                }
            }

            mediaRecorder.start(1000) // Collect 1 second of data at a time
        }
    }

    const gotLocalIceCandidateOffer = (event: RTCPeerConnectionIceEvent) => {
        console.log("event: ", event)
        if (!event.candidate) {
            const offer = localPeerConnection.localDescription
            sendWsMessage("send_offer", {
                channelName: searchParams.get("channelName"),
                userName: searchParams.get("userName"),
                sdp: offer,
            })
        }
    }

    const setupDevice = () => {
        ;(navigator.mediaDevices?.getUserMedia
            ? navigator.mediaDevices.getUserMedia({ audio: true, video: true })
            : new Promise<MediaStream>((resolve, reject) => {
                  ;(navigator as any).getUserMedia({ audio: true, video: true }, resolve, reject)
              })
        )
            .then((stream: MediaStream) => {
                const localPlayer = document.getElementById("localPlayer") as HTMLVideoElement | null
                if (localPlayer) {
                    localPlayer.srcObject = stream
                }
                localStream = stream

                // Start recording the local stream
                const options = {
                    mimeType: "video/webm",
                    videoBitsPerSecond: 2500000,
                    audioBitsPerSecond: 128000,
                }

                try {
                    mediaRecorder = new MediaRecorder(stream, options)
                } catch (e) {
                    console.error("Error creating MediaRecorder:", e)
                    mediaRecorder = new MediaRecorder(stream)
                }

                let chunks: Blob[] = []
                let startTime = Date.now()
                let chunkCount = 0

                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        chunks.push(e.data)
                        const duration = (Date.now() - startTime) / 1000

                        // Send chunk every 5 seconds
                        if (duration >= 5) {
                            const blob = new Blob(chunks, { type: "video/webm" })
                            const reader = new FileReader()
                            reader.onloadend = () => {
                                const base64data = reader.result as string
                                const cleanBase64 = base64data.split(",")[1]

                                // Create a new blob with WebM header for each chunk
                                const webmHeader = new Uint8Array([
                                    0x1a,
                                    0x45,
                                    0xdf,
                                    0xa3, // EBML Header
                                    0x42,
                                    0x86,
                                    0x81,
                                    0x01, // DocType
                                    0x42,
                                    0xf7,
                                    0x81,
                                    0x01, // DocTypeVersion
                                    0x42,
                                    0xf2,
                                    0x81,
                                    0x04, // DocTypeReadVersion
                                    0x42,
                                    0xf3,
                                    0x81,
                                    0x01, // DocTypeMaxIDLength
                                    0x42,
                                    0x82,
                                    0x84,
                                    0x77,
                                    0x65,
                                    0x62,
                                    0x6d, // DocTypeExtension
                                    0x18,
                                    0x53,
                                    0x80,
                                    0x67, // Segment
                                    0x01,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00,
                                    0x00, // Segment Size (unknown)
                                ])

                                const chunkBuffer = Buffer.from(cleanBase64, "base64")
                                const finalBuffer = Buffer.concat([webmHeader, chunkBuffer])
                                const finalBase64 = finalBuffer.toString("base64")

                                sendWsMessage("video_chunk", {
                                    channelName: searchParams.get("channelName"),
                                    userName: searchParams.get("userName"),
                                    chunk: finalBase64,
                                    timestamp: new Date().toISOString(),
                                    duration: duration,
                                    chunkNumber: chunkCount++,
                                })
                            }
                            reader.readAsDataURL(blob)

                            // Reset for next chunk
                            chunks = []
                            startTime = Date.now()
                        }
                    }
                }

                // Start recording
                mediaRecorder.start(1000) // Collect data every second

                ws.current?.send(
                    JSON.stringify({
                        type: "join",
                        body: {
                            channelName: searchParams.get("channelName"),
                            userName: searchParams.get("userName"),
                        },
                    })
                )
                setupPeerConnection()
            })
            .catch((err: any) => {
                console.log("err: ", err)
            })
    }

    // Add cleanup function
    useEffect(() => {
        return () => {
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop()
            }
            if (localStream) {
                localStream.getTracks().forEach((track) => track.stop())
            }
        }
    }, [])

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6">
            <Card className="w-full max-w-6xl flex flex-col md:flex-row gap-8 p-8 shadow-2xl rounded-2xl bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm border-0">
                <div className="flex-1 flex flex-col items-center gap-6">
                    <div className="w-full max-w-md">
                        <h2 className="text-2xl font-semibold text-blue-700 dark:text-blue-400 mb-4 text-center">
                            AI Assistant
                        </h2>
                        <div className="relative aspect-video rounded-xl overflow-hidden shadow-lg">
                            <AIAnimation isTalking={true} isThinking={false} />
                        </div>
                    </div>
                </div>
                <div className="flex-1 flex flex-col items-center gap-6">
                    <div className="w-full max-w-md">
                        <h2 className="text-2xl font-semibold text-blue-700 dark:text-blue-400 mb-4 text-center">
                            Video Call
                        </h2>
                        <div className="relative aspect-video rounded-xl overflow-hidden shadow-lg bg-black">
                            <video id="peerPlayer" autoPlay className="w-full h-full object-cover" />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-300">
                                <div className="absolute bottom-4 left-4 right-4 flex justify-center gap-4">
                                    <Button
                                        variant="outline"
                                        className="bg-red-500/90 hover:bg-red-600 text-white border-0 shadow-lg"
                                        onClick={handleEndCall}
                                    >
                                        End Call
                                    </Button>
                                    <Button
                                        variant="outline"
                                        className={`${
                                            isMuted
                                                ? "bg-red-500/90 hover:bg-red-600"
                                                : "bg-white/90 hover:bg-white"
                                        } text-gray-800 border-0 shadow-lg`}
                                        onClick={handleMute}
                                    >
                                        {isMuted ? "Unmute" : "Mute"}
                                    </Button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </Card>
        </div>
    )
}
