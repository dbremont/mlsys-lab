import math
import random
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# -----------------------------------------
# 1. NEURAL NETWORK LOGIC
# -----------------------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

dataset = [
    ([0, 0], 0),
    ([1, 0], 0),
    ([0, 1], 0),
    ([1, 1], 1)
]

def rand():
    return random.uniform(-1, 1)

# Global State
latest_state = {}
STOP_REQUESTED = False # Flag for stopping

def train_network():
    global latest_state, STOP_REQUESTED
    
    # Initialize Weights
    w11, w12, w21, w22 = rand(), rand(), rand(), rand()
    v1, v2 = rand(), rand()
    b1, b2, b3 = rand(), rand(), rand()
    
    learning_rate = 0.5
    epochs = 20000 # Target epochs

    print(f"Training started. Connect via browser to monitor.")

    for epoch in range(epochs):
        # Check stop flag
        if STOP_REQUESTED:
            print("Training stop requested by user.")
            break

        total_loss = 0
        batch_predictions = []

        for inputs, y in dataset:
            x1, x2 = inputs[0], inputs[1]

            # FORWARD PASS
            z1 = w11 * x1 + w12 * x2 + b1
            a1 = sigmoid(z1)
            z2 = w21 * x1 + w22 * x2 + b2
            a2 = sigmoid(z2)
            z3 = v1 * a1 + v2 * a2 + b3
            y_hat = sigmoid(z3)

            # STORE PREDICTION
            batch_predictions.append({
                "input": inputs, "target": y, "output": y_hat
            })

            # LOSS
            loss = -(y * math.log(y_hat + 1e-9) + (1 - y) * math.log(1 - y_hat + 1e-9))
            total_loss += loss

            # BACKPROPAGATION
            delta3 = y_hat - y
            dv1, dv2, db3 = delta3 * a1, delta3 * a2, delta3
            
            delta1 = (v1 * delta3) * sigmoid_derivative(z1)
            delta2 = (v2 * delta3) * sigmoid_derivative(z2)

            dw11, dw12 = delta1 * x1, delta1 * x2
            dw21, dw22 = delta2 * x1, delta2 * x2
            db1, db2 = delta1, delta2

            # UPDATE WEIGHTS
            v1 -= learning_rate * dv1
            v2 -= learning_rate * dv2
            b3 -= learning_rate * db3
            w11 -= learning_rate * dw11
            w12 -= learning_rate * dw12
            w21 -= learning_rate * dw21
            w22 -= learning_rate * dw22
            b1 -= learning_rate * db1
            b2 -= learning_rate * db2

        # UPDATE GLOBAL STATE FOR SERVER
        latest_state = {
            "epoch": epoch,
            "loss": total_loss,
            "weights": {
                "w11": w11, "w12": w12, "w21": w21, "w22": w22,
                "v1": v1, "v2": v2
            },
            "predictions": batch_predictions
        }
        
        # Small delay to prevent CPU lock and make visualization smooth
        time.sleep(0.02)

    # Loop finished (either completed or stopped)
    if not STOP_REQUESTED:
        print("Training completed naturally.")
    else:
        # Send final state update to indicate stopped
        pass

# -----------------------------------------
# 2. SERVER IMPLEMENTATION (SSE + REST)
# -----------------------------------------

class SSEHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*') 
            self.end_headers()
            
            try:
                while True:
                    if STOP_REQUESTED:
                        # Signal client that we are done
                        data = json.dumps({"status": "stopped"})
                        self.wfile.write(f"data: {data}\n\n".encode('utf-8'))
                        break
                        
                    if latest_state:
                        data = json.dumps(latest_state)
                        msg = f"data: {data}\n\n"
                        self.wfile.write(msg.encode('utf-8'))
                        self.wfile.flush()
                    time.sleep(0.1)
            except (ConnectionResetError, BrokenPipeError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        global STOP_REQUESTED
        if self.path == '/stop':
            print("Received STOP request.")
            STOP_REQUESTED = True
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = json.dumps({"message": "Training stopped"})
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass # Silence console

def run_server():
    server = HTTPServer(('localhost', 8000), SSEHandler)
    print("Server running on http://localhost:8000")
    server.serve_forever()

# -----------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------

if __name__ == "__main__":
    # 1. Start Server in Background Thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # 2. Run Training
    train_network()
    
    # Keep alive briefly to flush final buffers
    time.sleep(1)