import socket
import os
import time
from pathlib import Path
import argparse
import threading
import select

class LuckGenerator:
    def __init__(self, socket_path="/tmp/luck.sock", delay=0.01, verbose=False):
        self.socket_path = socket_path
        self.delay = delay
        self.verbose = verbose
        self.sock = None
        self.client = None
        self.running = True
        
        # Time tracking
        self.time_diff_ns_last = 0
        self.total_generated = 0
        self.total_sent = 0
        self.start_time = time.time()

    def setup_socket(self):
        if Path(self.socket_path).exists():
            os.unlink(self.socket_path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.socket_path)
        self.sock.listen(1)
        self.sock.setblocking(False)
        print(f"Luck socket ready at {self.socket_path}")

    def handle_connections(self):
        while self.running:
            try:
                ready, _, _ = select.select([self.sock], [], [], 0.1)
                if ready:
                    client, _ = self.sock.accept()
                    if self.client:
                        self.client.close()
                    self.client = client
                    print("New client connected!")
            except Exception as e:
                if self.verbose:
                    print(f"Connection handling: {e}")
                time.sleep(0.1)

    def generate_luck(self):
        # Get start time in nanoseconds
        start_time_ns = time.perf_counter_ns()
        
        # Sleep for specified delay
        time.sleep(self.delay)
        
        # Get end time and calculate difference
        end_time_ns = time.perf_counter_ns()
        time_diff_ns = end_time_ns - start_time_ns
        
        # If this is the first measurement, store it and return None
        if self.time_diff_ns_last == 0:
            self.time_diff_ns_last = time_diff_ns
            return None
            
        # Calculate the difference between current and last measurement
        time_diff_diff_ns = time_diff_ns - self.time_diff_ns_last
        self.time_diff_ns_last = time_diff_ns
        
        return abs(time_diff_diff_ns)

    def run(self):
        try:
            self.setup_socket()
            
            # Start connection handler
            conn_thread = threading.Thread(target=self.handle_connections)
            conn_thread.daemon = True
            conn_thread.start()

            print("Generating luck values (Ctrl+C to stop)...")
            
            # Main loop
            while self.running:
                # Generate luck value
                luck = self.generate_luck()
                if luck is not None:  # Skip first measurement
                    self.total_generated += 1
                    
                    # Show stats in verbose mode
                    if self.verbose and self.total_generated % 100 == 0:
                        elapsed = time.time() - self.start_time
                        rate = self.total_generated / elapsed
                        print(f"\rRate: {rate:.1f} luck/sec | Generated: {self.total_generated} | Sent: {self.total_sent} | Current: {luck:.3f}ns", end="", flush=True)
                    
                    # Check if it's a "lucky" value (threshold from original code)
                    if luck < 6000:  # Using the threshold from original code
                        if self.verbose:
                            print(f"\nLucky value found: {luck:.3f}ns")
                        
                        # Send to client if connected
                        if self.client:
                            try:
                                luck_str = f"{luck:.3f}\n"
                                self.client.send(luck_str.encode())
                                self.total_sent += 1
                                if self.verbose:
                                    print(f"Value sent to client: {luck:.3f}ns")
                            except (BrokenPipeError, socket.error):
                                print("\nClient disconnected")
                                self.client = None
                    
        except KeyboardInterrupt:
            print("\nShutting down luck generator...")
        finally:
            self.running = False
            if self.client:
                self.client.close()
            if self.sock:
                self.sock.close()
            if Path(self.socket_path).exists():
                os.unlink(self.socket_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate luck values using nanosecond timing differences')
    parser.add_argument('-v', '--verbose', action='store_true', 
                      help='Show detailed output including luck values')
    parser.add_argument('-d', '--delay', type=float, default=0.01,
                      help='Base delay time in seconds (default: 0.01)')
    parser.add_argument('-s', '--socket', type=str, default='/tmp/luck.sock',
                      help='Unix socket path (default: /tmp/luck.sock)')
    
    args = parser.parse_args()
    
    generator = LuckGenerator(socket_path=args.socket, 
                            delay=args.delay,
                            verbose=args.verbose)
    generator.run()

    