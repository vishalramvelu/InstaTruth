from moviepy import VideoFileClip
import datetime
import yt_dlp
import whisper
import os

model = whisper.load_model('small')

def standardize_fn(name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{name}_{timestamp}.mp4"

def download_tt(url=None):
    if url == None:
        return None
    
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = f"tiktok_{timestamp}.%(ext)s"
        
        ydl_opts = {
            'format': 'best',
            'outtmpl': output_filename,
            'ignoreerrors': True,
            'no_warnings': False,
            'quiet': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                filename = ydl.prepare_filename(info)
                actual_filename = filename.split('.')[0] + '.mp4'
                
                # rename file if it doesn't have .mp4 extension
                if os.path.exists(filename) and not filename.endswith('.mp4'):
                    os.rename(filename, actual_filename)
                    return actual_filename
                return filename
            else:
                print(f'[download_tt]: no video found')
                return None
    except Exception as e:
        print(f'[download_tt]: error downloading video: {e}')
        return None

def download_reel(url):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = f"instagram_{timestamp}.%(ext)s"
        
        ydl_opts = {
            'format': 'best',
            'outtmpl': output_filename,
            'ignoreerrors': True,
            'no_warnings': False,
            'quiet': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                filename = ydl.prepare_filename(info)
                actual_filename = filename.split('.')[0] + '.mp4'
                
                # rename file if it doesn't have .mp4 extension
                if os.path.exists(filename) and not filename.endswith('.mp4'):
                    os.rename(filename, actual_filename)
                    return actual_filename
                return filename
            else:
                return None
    except Exception as e:
        print(f'[download_reel]: error downloading video: {e}')
        return None

def mp4towav(filename):
    try:
        if filename == None:
            print(f'[mp4towav]: no mp4 provided')
            return None

        if not filename.endswith('.mp4'):
            print(f'[mp4towav]: no mp4 provided')
            return None

        wav_filename = filename.replace('.mp4', '.wav')
        video = VideoFileClip(filename)
        audio = video.audio
        audio.write_audiofile(wav_filename)
        audio.close()
        video.close()
        os.remove(filename)

        return wav_filename
    except Exception as e:
        print(f'[mp4towav]: error converting mp4 to wav: {e}')
        return None
    
def parse_audio(filename):
    try:
        result = model.transcribe(filename)
        return result['text']
    except Exception as e:
        print(f'[parse_audio]: error parsing audio: {e}')
        return None

def tt_to_text(url=None):
    if url == None:
        print(f'[tt_to_text]: no url provided')
        return None

    vidname = download_tt(url)
    if vidname == None:
        print(f'[tt_to_text]: no video found')
        return None

    wavname = mp4towav(vidname)
    if wavname == None:
        print(f'[tt_to_text]: no audio found')
        return None

    text = parse_audio(wavname)
    print(f'[tt_to_text]: {text}')
    os.remove(wavname)

    return text

def reel_to_text(url=None):
    if url == None:
        print(f'[reel_to_text]: no url provided')
        return None

    vidname = download_reel(url)
    if vidname == None:
        print(f'[reel_to_text]: no video found')
        return None

    wavname = mp4towav(vidname)
    if wavname == None:
        print(f'[reel_to_text]: no audio found')
        return None

    text = parse_audio(wavname)
    print(f'[reel_to_text]: {text}')
    os.remove(wavname)

    return text

if __name__ == "__main__":
    tturl = 'https://www.tiktok.com/@realdonaldtrump/video/7432520023958146335?is_from_webapp=1&sender_device=pc'
    reelurl = 'https://www.instagram.com/reel/DIPiq-8MFlf/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA=='
    tttext = tt_to_text(tturl)
    reeltext = reel_to_text(reelurl)
    print(f'[Tiktok]: {tttext}')
    print(f'[Instagram]" {reeltext}')