from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins.google.beta.realtime.realtime_api import AudioTranscriptionConfig
from livekit.plugins.google.beta.realtime import RealtimeModel
# from livekit.plugins import noise_cancellation
import os, cv2, asyncio, datetime, numpy as np

# Load environment variables from .env file
load_dotenv()  # read LIVEKIT_*, GOOGLE_API_KEY

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # initialize realtime model and session
    rt_model = RealtimeModel(
        model="gemini-2.0-flash-exp",
        voice="Puck",
        temperature=0.8,
        instructions="You are an interviewer. You will be speaking to your possible future candidate for the role ",
        api_key=os.getenv("GOOGLE_API_KEY"),
        # input_audio_transcription=AudioTranscriptionConfig(),
        output_audio_transcription=AudioTranscriptionConfig(),
    )
    rt_session = rt_model.session()
    session = AgentSession(llm=rt_model)
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

        await session.start(
            room=ctx.room,
            agent=Assistant(),
            room_input_options=input_opts,
        )
    except Exception as e:
        print(f"[ERROR] Agent session crashed")

    # subscribe to incoming video tracks
    async def handle_track(track: rtc.Track):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            video_stream = rtc.VideoStream(track)
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
                        return  # or skip the frame
                    arr = arr.reshape((height, width, 4))
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                    cv2.imshow("Remote Video", bgr)
                    cv2.waitKey(1)
                except ValueError as e:
                    print(f"[ERROR] Frame reshape failed: {e}")
                    return
            await video_stream.aclose()

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(handle_track(track))

    # publish local camera video
    WIDTH, HEIGHT = 640, 480
    source = rtc.VideoSource(WIDTH, HEIGHT)
    track = rtc.LocalVideoTrack.create_video_track("camera", source)
    opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await ctx.agent.publish_track(track, opts)

    # publish local microphone audio for transcription
    # AudioSource requires sample_rate and num_channels
    audio_source = rtc.AudioSource(24000, 1)
    mic_track = rtc.LocalAudioTrack.create_audio_track("microphone", audio_source)
    mic_opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await ctx.agent.publish_track(mic_track, mic_opts)

    async def _publish_camera():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # show local camera feed
            cv2.imshow("Local Camera", frame)
            cv2.waitKey(1)
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            vf = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, rgba.tobytes())
            source.capture_frame(vf)
            await asyncio.sleep(0)
        cap.release()

    asyncio.create_task(_publish_camera())

    def log_chat(role, text):
        with open("chat_history.txt", "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {role}: {text}\n")

    reply = await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )
    log_chat("Assistant", reply)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))