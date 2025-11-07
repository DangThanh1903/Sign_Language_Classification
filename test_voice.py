import asyncio, edge_tts, os

async def main():
    text  = "Xin chào! Đây là kiểm tra giọng nói tiếng Việt."
    voice = "vi-VN-HoaiMyNeural"   # hoặc "vi-VN-NamMinhNeural"
    out   = "say_vi.mp3"
    await edge_tts.Communicate(text, voice=voice).save(out)
    os.startfile(out)  # mở bằng trình phát mặc định trên Windows

asyncio.run(main())
