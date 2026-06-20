from __future__ import annotations
from pydantic import BaseModel

class Detection(BaseModel):
    federated: float
    siloed: float

class Bank(BaseModel):
    id: str; name: str; customers: int
    detection: Detection; poisoned: bool = False

class RegimeCounters(BaseModel):
    fraudPreventedGbp: float; timeToDetectHours: float; victims: int; lostGbp: float

class Counters(BaseModel):
    federated: RegimeCounters; siloed: RegimeCounters

class State(BaseModel):
    round: int; running: bool; banks: list[Bank]; counters: Counters
    campaignActive: bool; attackActive: bool; customerRecordsTransmitted: int = 0
