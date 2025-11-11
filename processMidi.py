import torch
import torchaudio
import sys
#from transformers import AutoModelForCasualLM
import torchaudio.functional as F



inFile = sys.argv[1]
try:
    # Load your audio file
    audio, sr = torchaudio.load(inFile)
    print(f'Loaded audio: shape={audio.shape}, sample_rate={sr}')
    
    # Load RAVE model
    model = torch.jit.load(sys.argv[3])
    model.eval()

    target_sr = 48000  # Try this first, then 24000 if it fails
    
    # Resample if needed
    if sr != target_sr:
        print(f'Resampling from {sr}Hz to {target_sr}Hz')
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    # Ensure mono and add batch dimension if needed
    if audio.shape[0] > 1:  # Convert stereo to mono
        audio = torch.mean(audio, dim=0, keepdim=True)

   # Apply lowpass filter directly
    audio = F.lowpass_biquad(
        audio, 
        sample_rate=target_sr,
        cutoff_freq=600,
        Q=0.707
    )

    # RMS normalization (often works better for neural audio)
    rms = torch.sqrt(torch.mean(audio**2))
    audio = audio / rms * 0.1  # Try different target RMS values

    # DC offset removal
    audio = audio - audio.mean()
    
    # Add batch dimension: (channels, samples) -> (batch, channels, samples)
    if len(audio.shape) == 2:
        audio = audio.unsqueeze(0)


    
    print(f'Final input shape: {audio.shape}')
    
    # Process through RAVE
    with torch.no_grad():
        output = model.forward(audio)
        print(f'Output shape: {output.shape}')

        # # Save output
        if len(output.shape) == 3:
            output = output.squeeze(0)
        
        torchaudio.save(sys.argv[2], output.cpu(), target_sr)
        print(f'Saved {sys.argv[2]}')
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()