from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class OpenEnvHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "running"}')

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        dummy_response = {
            "observation": "System initialized",
            "reward": 0.0,
            "done": False,
            "info": {"status": "ok"}
        }
        self.wfile.write(json.dumps(dummy_response).encode('utf-8'))

print("Starting OpenEnv submission server on port 7860...")
HTTPServer(('0.0.0.0', 7860), OpenEnvHandler).serve_forever()