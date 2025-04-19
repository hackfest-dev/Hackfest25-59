from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins.google.beta.realtime.realtime_api import AudioTranscriptionConfig
from livekit.plugins.google.beta.realtime import RealtimeModel
# from livekit.plugins import noise_cancellation
import os, cv2, asyncio, datetime, numpy as np
import time
import sys

# Load environment variables from .env file
load_dotenv()  # read LIVEKIT_*, GOOGLE_API_KEY

# Validate required environment variables
required_env_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "GOOGLE_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"[ERROR] Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

async def entrypoint(ctx: agents.JobContext):
    # Maximum number of connection retries
    MAX_RETRIES = 3
    retry_count = 0
    connection_successful = False
    
    while retry_count < MAX_RETRIES and not connection_successful:
        try:
            print(f"[INFO] Connecting to LiveKit room (attempt {retry_count + 1}/{MAX_RETRIES})...")
            await ctx.connect()
            connection_successful = True
            print("[INFO] Successfully connected to LiveKit room")
        except Exception as e:
            retry_count += 1
            print(f"[ERROR] Failed to connect: {str(e)}")
            if retry_count < MAX_RETRIES:
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"[INFO] Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print("[ERROR] Maximum connection attempts reached. Exiting.")
                return
    
    if not connection_successful:
        return

    # initialize realtime model and session
    try:
        rt_model = RealtimeModel(
            model="gemini-2.0-flash-exp",
            voice="Puck",
            temperature=0.8,
            instructions="You are an experienced HR professional whose sole mission right now is to interview a candidate for the role of Web Developer in a completely natural, human‑to‑human conversation. Begin by greeting the candidate warmly, introducing yourself in a friendly, approachable tone, and inviting them to tell you about their journey into web development and what drew them to this role. Ask open‑ended questions that encourage storytelling about specific projects, problem‑solving approaches, collaboration experiences and performance optimizations. Listen actively, respond with empathy, follow up on anything interesting or unclear, and probe for concrete examples—how they debugged tricky JavaScript issues, why they chose one framework over another, what they learned from code reviews they led. Keep a forward‑thinking focus by asking how they stay current with new technologies and envision their growth as a developer, while maintaining a healthy dose of skepticism by requesting evidence and details for any claims. Mirror natural human pacing with thoughtful pauses and conversational language that balances professionalism with warmth. After you have asked roughly ten substantive questions, wrap up the interview gracefully: thank the candidate for their time, invite any final questions they may have, and explain the next steps in the process.",
            api_key=os.getenv("GOOGLE_API_KEY"),
            # input_audio_transcription=AudioTranscriptionConfig(),
            output_audio_transcription=AudioTranscriptionConfig(),
        )
        rt_session = rt_model.session()
        session = AgentSession(llm=rt_model)

        import heapq
        import traceback
        from livekit.agents.voice import agent_activity

        # Monkey-patch _main_task of agent_activity to add error handling
        original_main_task = agent_activity.AgentActivity._main_task

        async def patched_main_task(self):
            try:
                await original_main_task(self)
            except TypeError as e:
                if "'<' not supported between instances of 'SpeechHandle'" in str(e):
                    print("[ERROR] Heap comparison failed in speech queue. Resetting _speech_q to recover...")
                    self._speech_q = []
                else:
                    print("[ERROR] Unexpected TypeError in _main_task:")
                    traceback.print_exc()
            except Exception as e:
                print("[ERROR] Unhandled exception in _main_task:")
                traceback.print_exc()

        agent_activity.AgentActivity._main_task = patched_main_task

    except Exception as e:
        print(f"[ERROR] Failed to initialize model or session: {str(e)}")
        return
        
    # register transcription handler on session emitter
    def on_audio_transcription(event):
        # extract text and source role
        text = getattr(event, "transcript", getattr(event, "text", ""))
        role = getattr(event, "source", "User")
        print(f"[DEBUG] transcription: {role}: {text}")
        log_chat(role, text)
    rt_session.on("input_audio_transcription_completed", on_audio_transcription)

    # enable audio input for transcription
    input_opts = RoomInputOptions(audio_enabled=True, video_enabled=True,
    #  noise_cancellation=noise_cancellation.BVC()
     )

    try:
        print("[INFO] Starting agent session...")
        await session.start(
            room=ctx.room,
            agent=Assistant(),
            room_input_options=input_opts,
        )
        print("[INFO] Agent session started successfully")
    except Exception as e:
        print(f"[ERROR] Agent session crashed: {str(e)}")
        return

    # subscribe to incoming video tracks
    async def handle_track(track: rtc.Track):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            video_stream = rtc.VideoStream(track)
            try:
                async for event in video_stream:
                    # handle incoming video frames
                    frame = event.frame
                    rgba = frame.data
                    height, width = frame.height, frame.width
                    try:
                        arr = np.frombuffer(rgba, dtype=np.uint8)
                        expected_size = height * width * 4
                        if arr.size != expected_size:
                            print(f"[ERROR] Frame size mismatch: got {arr.size}, expected {expected_size}")
                            continue  # Skip the frame instead of returning
                        arr = arr.reshape((height, width, 4))
                        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                        cv2.imshow("Remote Video", bgr)
                        cv2.waitKey(1)
                    except ValueError as e:
                        print(f"[ERROR] Frame reshape failed: {e}")
                        continue
            except Exception as e:
                print(f"[ERROR] Video stream processing error: {str(e)}")
            finally:
                await video_stream.aclose()

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(handle_track(track))

    try:
        # publish local camera video
        WIDTH, HEIGHT = 640, 480
        source = rtc.VideoSource(WIDTH, HEIGHT)
        track = rtc.LocalVideoTrack.create_video_track("camera", source)
        opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        await ctx.agent.publish_track(track, opts)
        print("[INFO] Published local video track")
        
        # publish local microphone audio for transcription
        # AudioSource requires sample_rate and num_channels
        audio_source = rtc.AudioSource(24000, 1)
        mic_track = rtc.LocalAudioTrack.create_audio_track("microphone", audio_source)
        mic_opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await ctx.agent.publish_track(mic_track, mic_opts)
        print("[INFO] Published local audio track")
    except Exception as e:
        print(f"[ERROR] Failed to publish local tracks: {str(e)}")
        # Continue execution - we might still be able to receive remote tracks

    async def _publish_camera():
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[ERROR] Failed to open camera")
                return
                
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            print("[INFO] Camera opened successfully")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from camera")
                    # Try to reopen camera after brief delay
                    await asyncio.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("[ERROR] Failed to reopen camera, stopping camera publishing")
                        break
                    continue
                    
                # show local camera feed
                cv2.imshow("Local Camera", frame)
                cv2.waitKey(1)
                rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                vf = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, rgba.tobytes())
                source.capture_frame(vf)
                await asyncio.sleep(0.01)  # Small sleep to prevent CPU hogging
        except Exception as e:
            print(f"[ERROR] Camera publishing error: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            print("[INFO] Camera released")

    camera_task = asyncio.create_task(_publish_camera())

    def log_chat(role, text):
        try:
            with open("chat_history.txt", "a", encoding="utf-8") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] {role}: {text}\n")
        except Exception as e:
            print(f"[ERROR] Failed to log chat: {str(e)}")

    try:
        reply = await session.generate_reply(
            instructions="You are an experienced HR professional taking interviews. Greet the candidate."
        )
        log_chat("Assistant", reply)
        print("[INFO] Generated initial reply")
    except Exception as e:
        print(f"[ERROR] Failed to generate reply: {str(e)}")

    # Keep the session running until disconnected
    disconnected = asyncio.Event()
    def on_disconnected():
        print("[INFO] Disconnected from LiveKit room, will attempt to reconnect...")
        disconnected.set()
    ctx.room.on("disconnected", lambda: on_disconnected())
    try:
        while True:
            await asyncio.wait([asyncio.create_task(disconnected.wait())], return_when=asyncio.FIRST_COMPLETED)
            if disconnected.is_set():
                print("[INFO] Attempting to reconnect...")
                camera_task.cancel()
                try:
                    await camera_task
                except asyncio.CancelledError:
                    pass
                cv2.destroyAllWindows()
                # Re-run entrypoint logic for reconnection
                await entrypoint(ctx)
                return
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("[INFO] Session cancelled")
    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt detected")
    finally:
        camera_task.cancel()
        try:
            await camera_task
        except asyncio.CancelledError:
            pass
        cv2.destroyAllWindows()
        print("[INFO] Session ended")

if __name__ == "__main__":
    try:
        agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    except KeyboardInterrupt:
        print("[INFO] Program terminated by user")
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {str(e)}")