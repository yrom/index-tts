import os
import sys
import warnings
# Suppress warnings from tensorflow and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="IndexTTS Command Line")
    parser.add_argument("text", type=str, help="Text to be synthesized")
    parser.add_argument("-v", "--voice", type=str, required=True, help="Path to the audio prompt file (wav format)")
    parser.add_argument("-o", "--output_path", type=str, default="gen.wav", help="Path to the output wav file")
    parser.add_argument("-c", "--config", type=str, default="checkpoints/config.yaml", help="Path to the config file. Default is 'checkpoints/config.yaml'")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Path to the model directory. Default is 'checkpoints'")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 for inference if available")
    parser.add_argument("-f", "--force", action="store_true", default=False, help="Force to overwrite the output file if it exists")
    parser.add_argument("-d", "--device", type=str, default=None, help="Device to run the model on (cpu, cuda, mps)." )
    args = parser.parse_args()
    if len(args.text.strip()) == 0:
        print("ERROR: Text is empty.")
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.voice):
        print(f"Audio prompt file {args.voice} does not exist.")
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"Config file {args.config} does not exist.")
        parser.print_help()
        sys.exit(1)

    output_path = args.output_path
    if os.path.exists(output_path):
        if not args.force:
            print(f"ERROR: Output file {output_path} already exists. Use --force to overwrite.")
            parser.print_help()
            sys.exit(1)
        else:
            os.remove(output_path)
    
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install it first.")
        sys.exit(1)

    def auto_select_torch_devic():
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.mps.is_available():
            if torch.backends.mps.is_built():
                return "mps"
            warnings.warn(
                "WARNING: MPS not available because the current PyTorch install was not built with MPS enabled.",
                category=RuntimeWarning,
            )
        warnings.warn("WARNING: Running on CPU may be slow.", category=RuntimeWarning)
        return "cpu"
        
    if args.device is None:
        args.device = auto_select_torch_devic()
    if args.fp16 and not torch.is_autocast_enabled(args.device.split(":")[0]):
        warnings.warn(
            f"WARNING: AMP is disabled on the current device({args.device}). Falling back to FP32.",
            category=RuntimeWarning,
        )
        args.fp16 = False

    from indextts.infer import IndexTTS
    tts = IndexTTS(cfg_path=args.config, model_dir=args.model_dir, is_fp16=args.fp16, device=args.device)
    tts.infer(audio_prompt=args.voice, text=args.text.strip(), output_path=output_path)

if __name__ == "__main__":
    main()