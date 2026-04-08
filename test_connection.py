import cv2
import requests
from requests.auth import HTTPBasicAuth

# Test connection to OD-325HD camera
ip_address = "http://10.1.193.29"
username = "admin"
password = "admin"

print(f"Testing connection to {ip_address}")

# Test basic connectivity
try:
    response = requests.get(ip_address, timeout=5)
    print(f"✓ Camera is reachable: {response.status_code}")
except Exception as e:
    print(f"✗ Cannot reach camera: {e}")

# Test snapshot
try:
    snapshot_url = f"{ip_address.rstrip('/')}/snapshot.jpg"
    auth = HTTPBasicAuth(username, password)
    response = requests.get(snapshot_url, auth=auth, timeout=10)
    print(f"✓ Snapshot available: {response.status_code}")
except Exception as e:
    print(f"✗ Snapshot failed: {e}")

# Test MJPEG stream URLs
stream_urls = [
    f'{ip_address}/mjpeg.cgi',
    f'{ip_address}/cgi-bin/mjpg/video.mjpg',
    f'{ip_address}/video.mjpg',
    f'{ip_address}/stream.mjpg',
    f'{ip_address}:8080/video.mjpg',
    f'{ip_address}/axis-cgi/mjpg/video.cgi',
    f'{ip_address}/img/video.mjpeg',
    f'{ip_address}/videostream.cgi',
    f'{ip_address}/live.mjpg',
]

print("\nTesting MJPEG URLs:")
for i, url in enumerate(stream_urls, 1):
    print(f"[{i}/{len(stream_urls)}] Testing: {url}")
    try:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ SUCCESS: {url} - Frame shape: {frame.shape}")
                cap.release()
                break
            else:
                print(f"✗ Cannot read frame: {url}")
                cap.release()
        else:
            print(f"✗ Cannot open: {url}")
    except Exception as e:
        print(f"✗ Error with {url}: {str(e)[:50]}")

print("\nTest completed.")