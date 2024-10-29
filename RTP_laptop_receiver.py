import vlc
import time

# Create VLC instances for each stream
instance1 = vlc.Instance()
player1 = instance1.media_player_new()
media1 = instance1.media_new('rtp://@:5000')
player1.set_media(media1)

instance2 = vlc.Instance()
player2 = instance2.media_player_new()
media2 = instance2.media_new('rtp://@:5001')
player2.set_media(media2)

# Start playing
player1.play()
player2.play()

# Keep the script running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Stop playback
    player1.stop()
    player2.stop()
