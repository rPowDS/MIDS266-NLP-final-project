#!/usr/bin/env python3
"""
Example: Training BART on CNN/DailyMail
"""

from align_rerank.train_bart import main

if __name__ == "__main__":
    # Train with custom parameters
    main(
        output_dir="runs/my-bart-model",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=5e-5
    )
