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
from training.train import evaluate_model


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained ergonomic chair model')
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='./models/chair_ppo_final.zip',
        help='Path to trained model (default: ./models/chair_ppo_final.zip)'
    )
    
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes (default: 10)'
    )
    
    parser.add_argument(
        '--render',
        action='store_true',
        help='Enable rendering during evaluation'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Ergonomic Chair AI Evaluation")
    print("=" * 60)
    print(f"Model Path: {args.model_path}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Render: {args.render}")
    print("=" * 60)
    
    episode_rewards, episode_lengths = evaluate_model(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        render=args.render
    )


if __name__ == '__main__':
    main()
