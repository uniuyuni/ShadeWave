from enum import Enum

class EffectMode(int, Enum):
    PREVIEW = 0
    LOUPE = 1
    EXPORT = 2

class ExecutionMode(int, Enum):
    SYNC = 0
    ASYNC = 1
    BLOCKING = 2

# Status of upstream processing
class PipelineStatus(int, Enum):
    PREVIEW = 0   # Processing or Preview quality
    COMPLETE = 1  # High quality complete
