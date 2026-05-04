# argd/training
try:
    from .trainer import HIATrainer
except Exception:
    HIATrainer = None

__all__ = ['HIATrainer']
