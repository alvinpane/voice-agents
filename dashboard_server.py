#!/usr/bin/env python3
import http.server
import socketserver
import json
import os
from urllib.parse import urlparse

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for the IVR dashboard"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Route for listing result files
        if path == '/list-results':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            try:
                result_files = []
                results_dir = 'results'
                
                if os.path.exists(results_dir) and os.path.isdir(results_dir):
                    result_files = os.listdir(results_dir)
                
                self.wfile.write(json.dumps(result_files).encode())
            except Exception as e:
                self.wfile.write(json.dumps({
                    "error": str(e)
                }).encode())
            
            return
        
        # Serve the dashboard.html file as the root
        if path == '/' or path == '/index.html':
            self.path = '/dashboard.html'
        
        # Handle normal file serving
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def log_message(self, format, *args):
        """Custom logging for the server"""
        if '/list-results' not in args[0]:  # Don't log API calls to reduce noise
            http.server.SimpleHTTPRequestHandler.log_message(self, format, *args)

def run_server(port=8000):
    """
    Run the dashboard web server
    
    Args:
        port: Port number to serve on (default: 8000)
    """
    # Ensure the dashboard.html file exists
    if not os.path.exists('dashboard.html'):
        print("Warning: dashboard.html not found. Extracting from IVR dashboard...")
        try:
            # We'll look for the HTML content in this script file
            with open(__file__, 'r') as script_file:
                script_content = script_file.read()
                
                # Look for HTML content between markers (not implemented here)
                # In a real implementation, you might embed the HTML or have it as a separate file
                
                print("Error: dashboard.html could not be automatically extracted.")
                print("Please ensure the dashboard.html file exists in the current directory.")
                return
        except Exception as e:
            print(f"Error extracting dashboard: {e}")
            return
    
    # Create a web server
    handler = DashboardHandler
    httpd = socketserver.TCPServer(("", port), handler)
    
    print(f"Starting dashboard server at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    # Serve until interrupted
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="IVR Test Results Dashboard Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve the dashboard on")
    
    args = parser.parse_args()
    run_server(port=args.port)