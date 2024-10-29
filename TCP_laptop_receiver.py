import socket
import av
import cv2
import struct

# TCP Configuration
TCP_IP = "0.0.0.0"  # Bind to all available interfaces
TCP_PORT = 5005

# Set up TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_IP, TCP_PORT))
sock.listen(1)
print("Waiting for Jetson connection...")
connection, addr = sock.accept()
print(f"Connected to {addr}")

# Initialize codec
codec = av.CodecContext.create('h264', 'r')

while True:
    try:
        # Read the packet size first (4 bytes)
        packet_size_data = connection.recv(4)
        if not packet_size_data:
            break  # No data means the connection is closed

        # Unpack packet size
        packet_size = struct.unpack("I", packet_size_data)[0]

        # Receive the actual packet based on size
        packet_data = b""
        while len(packet_data) < packet_size:
            packet_data += connection.recv(packet_size - len(packet_data))

        # Decode the received packet
        packet = av.packet.Packet(packet_data)
        for frame in codec.decode(packet):
            frame_bgr = frame.to_ndarray(format='bgr24')
            cv2.imshow("Received Video", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error receiving or decoding frame: {e}")
        break

# Cleanup
cv2.destroyAllWindows()
connection.close()
sock.close()
