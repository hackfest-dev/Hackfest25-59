const app = require('express')();
const wss = require('./wss');

const HTTP_PORT = 4000;
const WEB_SOCKET_PORT = 8090

wss.init(WEB_SOCKET_PORT)

app.listen(HTTP_PORT, () => {
    console.log("Server is listening on port", HTTP_PORT);
})