"use client"

import type React from "react"

import {
  ControlBar,
  GridLayout,
  ParticipantTile,
  RoomAudioRenderer,
  useTracks,
  RoomContext,
  useConnectionState,
} from "@livekit/components-react"
import { Room, Track, ConnectionState } from "livekit-client"
import "@livekit/components-styles"
import { useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { Users, Video, Wifi, WifiOff } from "lucide-react"

export default function Page() {
  // TODO: get user input for room and name
  const room = "quickstart-room"
  const name = "quickstart-user"
  const [token, setToken] = useState("")
  const [isLoading, setIsLoading] = useState(true)
  const [roomInstance] = useState(
    () =>
      new Room({
        // Optimize video quality for each participant's screen
        adaptiveStream: true,
        // Enable automatic audio/video quality optimization
        dynacast: true,
      }),
  )

  useEffect(() => {
    let mounted = true
    setIsLoading(true)
    ;(async () => {
      try {
        const resp = await fetch(`/api/token?room=${room}&username=${name}`)
        const data = await resp.json()
        if (!mounted) return

        if (data.token) {
          setToken(data.token)
          console.log("Connecting to room at:", process.env.NEXT_PUBLIC_LIVEKIT_URL)
          await roomInstance.connect(process.env.NEXT_PUBLIC_LIVEKIT_URL!, data.token)

          // Immediately start camera + mic
          await roomInstance.localParticipant.setCameraEnabled(true)
          await roomInstance.localParticipant.setMicrophoneEnabled(true)
        }
      } catch (e) {
        console.error(e)
      } finally {
        if (mounted) {
          setIsLoading(false)
        }
      }
    })()

    return () => {
      mounted = false
      roomInstance.disconnect()
    }
  }, [roomInstance])

  if (isLoading || token === "") {
    return <LoadingState />
  }

  return (
    <RoomContext.Provider value={roomInstance}>
      <div data-lk-theme="default" className="flex flex-col h-screen bg-slate-50">
        <Header roomName={room} userName={name} />
        <div className="flex-1 relative">
          {/* Your custom component with basic video conferencing functionality. */}
          <MyVideoConference />
          {/* The RoomAudioRenderer takes care of room-wide audio for you. */}
          <RoomAudioRenderer />
        </div>
        {/* Controls for the user to start/stop audio, video, and screen share tracks */}
        <div className="bg-white border-t border-slate-200 shadow-sm">
          <ControlBar variation="minimal" className="!bg-transparent !p-4" />
        </div>
      </div>
    </RoomContext.Provider>
  )
}

function LoadingState() {
  return (
    <div className="flex items-center justify-center h-screen bg-slate-50">
      <Card className="w-full max-w-md">
        <CardContent className="pt-6">
          <div className="flex flex-col items-center space-y-4">
            <div className="rounded-full bg-primary/10 p-3">
              <Video className="h-8 w-8 text-primary" />
            </div>
            <h2 className="text-2xl font-bold text-center">Connecting to Meeting</h2>
            <p className="text-muted-foreground text-center">Please wait while we set up your video conference...</p>
            <div className="w-full space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4 mx-auto" />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function Header({ roomName, userName }: { roomName: string; userName: string }) {
  const connectionState = useConnectionState()

  return (
    <div className="bg-white border-b border-slate-200 py-3 px-4 flex items-center justify-between shadow-sm">
      <div className="flex items-center space-x-2">
        <Video className="h-5 w-5 text-primary" />
        <h1 className="font-semibold text-lg">LiveKit Meeting</h1>
      </div>

      <div className="flex items-center space-x-3">
        <Badge variant="outline" className="flex items-center gap-1">
          <Users className="h-3.5 w-3.5" />
          <span>{roomName}</span>
        </Badge>

        <ConnectionBadge state={connectionState} />
      </div>
    </div>
  )
}

function ConnectionBadge({ state }: { state: ConnectionState }) {
  let content

  switch (state) {
    case ConnectionState.Connected:
      content = (
        <Badge variant="success" className="flex items-center gap-1">
          <Wifi className="h-3.5 w-3.5" />
          <span>Connected</span>
        </Badge>
      )
      break
    case ConnectionState.Connecting:
      content = (
        <Badge variant="outline" className="flex items-center gap-1 bg-amber-50 text-amber-700 border-amber-200">
          <Wifi className="h-3.5 w-3.5" />
          <span>Connecting...</span>
        </Badge>
      )
      break
    case ConnectionState.Disconnected:
    case ConnectionState.Reconnecting:
    default:
      content = (
        <Badge variant="destructive" className="flex items-center gap-1">
          <WifiOff className="h-3.5 w-3.5" />
          <span>Disconnected</span>
        </Badge>
      )
  }

  return content
}

function MyVideoConference() {
  // `useTracks` returns all camera and screen share tracks. If a user
  // joins without a published camera track, a placeholder track is returned.
  const tracks = useTracks(
    [
      { source: Track.Source.Camera, withPlaceholder: true },
      { source: Track.Source.ScreenShare, withPlaceholder: false },
    ],
    { onlySubscribed: false },
  )

  return (
    <GridLayout tracks={tracks} className="h-full bg-slate-100">
      <EnhancedParticipantTile />
    </GridLayout>
  )
}

function EnhancedParticipantTile(props: React.ComponentProps<typeof ParticipantTile>) {
  return (
    <ParticipantTile
      {...props}
      className="rounded-lg overflow-hidden border border-slate-200 shadow-sm"
      style={{
        aspectRatio: "16/9",
        ...props.style,
      }}
    />
  )
}
