from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel

# SINGLE SOURCE OF TRUTH for the Veritas API contract.
#
# contract/schema.py (this file) is the canonical definition. contract/types.ts
# mirrors it for TypeScript consumers, and web/src/lib/types.ts mirrors that in
# turn for the web app. Field names are camelCase so the Python and TypeScript
# shapes are byte-identical over the wire (e.g. bankId, customerRecordsTransmitted).
# When the contract changes, change THIS file first, then propagate to types.ts.

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

class Provenance(BaseModel):
    """Per-round model bill-of-materials: members FedAvg'd into the global model
    vs the dropped poisoned update, and the resulting mean federated recall."""
    round: int
    contributors: list[str]
    rejected: list[str] = []
    globalRecall: float

Regime = Literal["federated", "siloed"]

class PredictRequest(BaseModel):
    """Payload sent to POST /predict. ``transaction`` is the whitelisted numeric
    feature map (camelCase keys mirroring the model FEATURE_ORDER); ``text`` is an
    optional free-text narrative for explanation/triage."""
    transaction: dict[str, Any]
    text: Optional[str] = None

class PredictResponse(BaseModel):
    """Response from POST /predict. Mirrors node/node/server/app.py exactly:
    ``confidence`` is a probability in the closed interval [0, 1]."""
    label: str
    confidence: float
    indicators: list[str]
    explanation: Optional[str] = None
