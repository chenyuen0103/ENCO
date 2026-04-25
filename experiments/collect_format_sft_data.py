#!/usr/bin/env python3
"""
Compatibility shim for the renamed SFT reasoning-data generator.

Use:
    python experiments/generate_reasoning.py ...
"""
from generate_reasoning import main


if __name__ == "__main__":
    main()
