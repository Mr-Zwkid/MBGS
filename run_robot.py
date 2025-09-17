
import sys
import subprocess
import os

def main():
    """
    This script is a Python version of the robot.sh bash script.
    It configures and runs the motionblender/train.py script based on a given scene.
    """
    # if len(sys.argv) < 2:
    #     print(f"Usage: python {sys.argv[0]} <scene> [extra_opts...]")
    #     sys.exit(1)

    # --- 1. Parse Arguments ---
    scene = 'ur5_ep1'
    iteration = 6000
    extra_opts = sys.argv[1:]

    # --- 2. Basic Configuration ---
    workdir = f"outputs/robot/{scene}"
    train_opts = ["--train-steps", str(iteration)]
    
    # --- 3. Scene-specific Configuration (replaces the first case statement) ---
    canoid = 0
    humanid = None
    cameras = ""

    if scene == "robot":
        canoid = 323
    elif scene == "microwave":
        canoid = 0
        humanid = 2
        cameras = "center left right"
    elif scene == "rope":
        canoid = 0
        humanid = 1
        cameras = "center"
    elif scene == "cloth":
        canoid = 0
        humanid = 1
        cameras = "center left right"

    # --- 4. Build Data Options ---
    data_opts = [
        "data:general",
        "--data.data-dir", f"./datasets/robot/{scene}",
        "--data.depth-type", "metric_depth",
        "--data.no-use-tracks",
        "--data.no-normalize-scene",
        "--data.use-median-filter",
        "--data.given-cano-t", str(canoid)
    ]
    if humanid is not None:
        data_opts.extend(["--data.mask-insts", str(humanid)])

    # --- 5. Build Camera Options ---
    camera_opts = []
    if cameras:
        camera_opts.extend(["--cameras", *cameras.split()])

    # --- 6. Build Basic Options (replaces the second case statement) ---
    basic_opts = [
        "--fg-only",
        "--loss.w-smooth-motion", "1.0",
        "--ctrl.stop-control-by-screen-steps", "4000",
        "--ctrl.stop-control-steps", "4000",
        "--lr.annealing-for", "means", "scales", "motions",
        "--lr.annealing-factor", "1e-2"
    ]
    basic_opts.extend(camera_opts)

    basic_opts.extend([
            "--loss.w-smooth-motion", "0.0",
            "--loss.w-sparse-link-assignment", "0.0",
            "--loss.w-minimal-movement-in-cano", "0.0",
            "--loss.w-rgb", "2.0"
        ])

    # if scene in ["robot", "ur5_ep1"]:
    #     basic_opts.extend([
    #         "--loss.w-smooth-motion", "0.0",
    #         "--loss.w-sparse-link-assignment", "0.0",
    #         "--loss.w-minimal-movement-in-cano", "0.0",
    #         "--loss.w-rgb", "2.0"
    #     ])
    # elif scene == "microwave":
    #     basic_opts.extend([
    #         "--loss.w-minimal-movement-in-cano", "0.0",
    #         "--loss.w-rgb", "2.0"
    #     ])
    # elif scene == "rope":
    #     basic_opts.extend([
    #         "--loss.w-minimal-movement-in-cano", "-1.0",
    #         "--loss.w-rgb", "4.0",
    #         "--loss.w-kp2d", "1.0",
    #         "--motion-init-batch-size", "-1",
    #         "--motion-pretrain-with-kp3d",
    #         "--no-motion-pretrain-with-means",
    #         "--motion-init-steps", "400",
    #         "--init-gamma", "50"
    #     ])
    # elif scene == "cloth":
    #     basic_opts.extend([
    #         "--loss.w-minimal-movement-in-cano", "-1.0",
    #         "--loss.w-rgb", "4.0",
    #         "--loss.w-length-reg", "0.5",
    #         "--loss.length-reg-names", "cloth",
    #         "--loss.w-arap", "0.3"
    #     ])

    # --- 7. Assemble and Run the Final Command ---
    final_command = [
        "python", "motionblender/train.py",
        "--work-dir", workdir,
        *basic_opts,
        *train_opts,
        *extra_opts,
        *data_opts
    ]

    # Print the command to be executed (similar to `set -x`)
    print("Executing command:")
    print(' '.join(final_command))
    print("-" * 20)

    # Execute the command
    try:
        subprocess.run(final_command, check=True)
    except FileNotFoundError:
        print(f"Error: 'python' command not found. Make sure Python is in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
