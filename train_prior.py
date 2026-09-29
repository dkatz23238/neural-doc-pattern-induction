"""Same training program for a laptop CPU and a later GPU job.

    python train_prior.py --device cpu --model-size 100k --steps 20000 --batch-size 32
"""

from prior.train import main

if __name__ == "__main__":
    main()
