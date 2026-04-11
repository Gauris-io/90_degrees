import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from env import TalentArbitrageEnv, Action

# Global environment instance
current_env = TalentArbitrageEnv(task_level="easy")

class OpenEnvHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "running"}')

    def do_POST(self):
        global current_env
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        # Handle Grader Reset (Critical: Catch the task level!)
        if '/reset' in self.path:
            task_level = data.get("task_level", "easy")
            current_env = TalentArbitrageEnv(task_level=task_level)
            obs = current_env.reset()
            response = {
                "observation": json.loads(obs.model_dump_json()),
                "reward": 0.0,
                "done": False,
                "info": {"status": "ok", "task_level": task_level}
            }
            
        # Handle Grader Step
        else:
            try:
                action = Action(**data)
            except Exception:
                action = Action(command="submit") # Safety fallback
                
            obs, reward, done, info = current_env.step(action)
            response = {
                "observation": json.loads(obs.model_dump_json()),
                "reward": float(reward),
                "done": bool(done),
                "info": info
            }

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

def main():
    print("Starting OpenEnv submission server on port 7860...")
    server = HTTPServer(('0.0.0.0', 7860), OpenEnvHandler)
    server.serve_forever()

if __name__ == '__main__':
    main()