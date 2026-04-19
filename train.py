#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Harley Wang (王华)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
import sys
import logging
from training.train import train_ppo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train ergonomic chair AI model')
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Total training timesteps (default: 100000)'
    )
    
    parser.add_argument(
        '--n-envs',
        type=int,
        default=4,
        help='Number of parallel environments (default: 4)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    
    parser.add_argument(
        '--n-steps',
        type=int,
        default=2048,
        help='Number of steps per update (default: 2048)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=10,
        help='Number of epochs per update (default: 10)'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor (default: 0.99)'
    )
    
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='GAE lambda (default: 0.95)'
    )
    
    parser.add_argument(
        '--ent-coef',
        type=float,
        default=0.01,
        help='Entropy coefficient (default: 0.01)'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        default=True,
        help='Use GPU if available (default: True)'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Force CPU usage'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory for logs (default: ./logs)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='./models',
        help='Directory for model checkpoints (default: ./models)'
    )
    
    parser.add_argument(
        '--load-path',
        type=str,
        default=None,
        help='Path to load existing model for continued training'
    )
    
    parser.add_argument(
        '--save-freq',
        type=int,
        default=10000,
        help='Save checkpoint frequency (default: 10000)'
    )
    
    args = parser.parse_args()
    
    use_gpu = args.use_gpu and not args.no_gpu
    
    logger.info("=" * 60)
    logger.info("Ergonomic Chair AI Training")
    logger.info("=" * 60)
    logger.info(f"Total Timesteps: {args.timesteps:,}")
    logger.info(f"Parallel Environments: {args.n_envs}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"N Steps: {args.n_steps}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"N Epochs: {args.n_epochs}")
    logger.info(f"Gamma: {args.gamma}")
    logger.info(f"GAE Lambda: {args.gae_lambda}")
    logger.info(f"Entropy Coef: {args.ent_coef}")
    logger.info(f"Use GPU: {use_gpu}")
    logger.info(f"Log Directory: {args.log_dir}")
    logger.info(f"Model Directory: {args.model_dir}")
    logger.info("=" * 60)
    
    try:
        model, callback, summary = train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            use_gpu=use_gpu,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            load_path=args.load_path,
            save_freq=args.save_freq
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Total episodes: {summary['total_episodes']}")
        logger.info(f"Mean reward: {summary['mean_reward']:.2f}")
        logger.info(f"Max reward: {summary['max_reward']:.2f}")
        logger.info(f"Min reward: {summary['min_reward']:.2f}")
        logger.info(f"Training time: {summary['training_time_minutes']:.1f} minutes")
        logger.info(f"Final model path: {summary['final_model_path']}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
