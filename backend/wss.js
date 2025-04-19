const WebSocket = require("ws")
const fs = require("fs")
const path = require("path")
const { Buffer } = require("buffer")

// Create videos directory if it doesn't exist
const videosDir = path.join(__dirname, "videos")
if (!fs.existsSync(videosDir)) {
    fs.mkdirSync(videosDir)
}

// WebM header bytes
const WEBM_HEADER = Buffer.from([
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

const init = (port) => {
    console.log("Websocket server initiated")
    const wss = new WebSocket.Server({ port })
    wss.on("connection", (socket) => {
        console.log("A client has been connected!")

        socket.on("error", console.error)
        socket.on("message", (message) => onMessage(wss, socket, message))
        socket.on("close", (message) => onClose(wss, socket, message))
    })
}

const channels = {}
const videoStreams = {} // Store video streams for each user

const send = (wsClient, type, body) => {
    wsClient.send(JSON.stringify({ type, body }))
}

const onMessage = (wss, socket, message) => {
    try {
        const parsedMessage = JSON.parse(message)
        const { type, body } = parsedMessage
        const { channelName, userName } = body

        switch (type) {
            case "join": {
                console.log("A user has joined")
                if (!channels[channelName]) {
                    channels[channelName] = {}
                }
                channels[channelName][userName] = socket
                const userNames = Object.keys(channels[channelName])
                send(socket, "joined", userNames)
                break
            }
            case "video_chunk": {
                if (!body.chunk || !body.timestamp || !body.duration || body.chunkNumber === undefined) {
                    console.error("Missing required fields in video chunk")
                    return
                }

                try {
                    // Convert base64 to buffer
                    const buffer = Buffer.from(body.chunk, "base64")

                    // Create a directory for this user if it doesn't exist
                    const userDir = path.join(videosDir, userName)
                    if (!fs.existsSync(userDir)) {
                        fs.mkdirSync(userDir)
                    }

                    // Create filename with timestamp, duration, and chunk number
                    const safeTimestamp = body.timestamp.replace(/[:.]/g, "-")
                    const duration = Math.round(body.duration)
                    const filename = `${safeTimestamp}_${duration}s_${body.chunkNumber}.webm`
                    const filepath = path.join(userDir, filename)

                    // Ensure the chunk has the WebM header
                    let chunkBuffer = buffer
                    // Check if the buffer starts with the WebM header
                    const headerLength = WEBM_HEADER.length
                    const hasHeader = buffer.slice(0, headerLength).equals(WEBM_HEADER)
                    if (!hasHeader) {
                        chunkBuffer = Buffer.concat([WEBM_HEADER, buffer])
                    }

                    // Write the chunk to file
                    fs.writeFileSync(filepath, chunkBuffer)

                    console.log(
                        `Saved video chunk ${body.chunkNumber} for ${userName} at ${body.timestamp} (${duration} seconds)`
                    )
                } catch (error) {
                    console.error("Error processing video chunk:", error)
                }
                break
            }
            case "quit": {
                if (channels[channelName]) {
                    channels[channelName][userName] = null
                    const userNames = Object.keys(channels[channelName])
                    if (!userNames.length) {
                        delete channels[channelName]
                    }
                }
                break
            }
            case "send_offer": {
                console.log("send_offer event received")
                const { sdp } = body
                const userNames = Object.keys(channels[channelName])
                userNames.forEach((uName) => {
                    if (uName.toString() !== userName.toString()) {
                        const wsClient = channels[channelName][uName]
                        send(wsClient, "offer_sdp_received", sdp)
                    }
                })
                break
            }
            case "send_answer": {
                const { sdp } = body
                const userNames = Object.keys(channels[channelName])
                userNames.forEach((uName) => {
                    if (uName.toString() !== userName.toString()) {
                        const wsClient = channels[channelName][uName]
                        send(wsClient, "answer_sdp_received", sdp)
                    }
                })
                break
            }
            case "send_ice_candidate": {
                const { candidate } = body
                const userNames = Object.keys(channels[channelName])
                userNames.forEach((uName) => {
                    if (uName.toString() !== userName.toString()) {
                        const wsClient = channels[channelName][uName]
                        send(wsClient, "ice_candidate_received", candidate)
                    }
                })
                break
            }
            default:
                break
        }
    } catch (error) {
        console.error("Error processing message:", error)
    }
}

const onClose = (wss, socket, message) => {
    console.log("onClose", message)
    Object.keys(channels).forEach((cname) => {
        Object.keys(channels[cname]).forEach((uid) => {
            if (channels[cname][uid] === socket) {
                delete channels[cname][uid]
            }
        })
    })
}

module.exports = { init }
