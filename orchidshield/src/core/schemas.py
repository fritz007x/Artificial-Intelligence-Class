"""Strict Pydantic schemas for every artifact that crosses a stage boundary of the
six-stage reasoning contract: PERCEPT -> PROPOSITION -> KNOWLEDGE -> INFERENCE -> ENTAILMENT -> ACTION.
All models forbid unknown fields so malformed inputs fail loudly."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------- raw inputs
Severity = Literal["low", "medium", "high", "critical"]
SEVERITY_NUM = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_ID = r"^[A-Za-z0-9_.:\-]{1,64}$"


class AlertIn(Strict):
    alert_id: str = Field(pattern=_ID)
    timestamp: datetime
    alert_type: str = Field(pattern=r"^[a-z_]{2,40}$")
    severity: Severity
    asset_id: str = Field(pattern=_ID)
    user_id: str | None = Field(default=None, pattern=_ID)
    src_ip: str | None = Field(default=None, max_length=45)
    indicators: list[str] = Field(default_factory=list, max_length=50)
    description: str = Field(default="", max_length=2000)


class AuthEventIn(Strict):
    event_id: str = Field(pattern=_ID)
    timestamp: datetime
    user_id: str = Field(pattern=_ID)
    asset_id: str = Field(pattern=_ID)
    src_ip: str | None = Field(default=None, max_length=45)
    geo: tuple[float, float] | None = None
    outcome: Literal["success", "failure"]
    mfa: bool = False


class AssetIn(Strict):
    asset_id: str = Field(pattern=_ID)
    hostname: str = Field(max_length=128)
    asset_type: Literal["workstation", "server", "database", "domain_controller", "firewall", "siem"]
    criticality: int = Field(ge=1, le=5)
    zone: Literal["dmz", "corp", "prod", "mgmt"]
    connected_to: list[str] = Field(default_factory=list)
    in_maintenance: bool = False
    business_unit: str = "unknown"
    edr_installed: bool = True


class UserIn(Strict):
    user_id: str = Field(pattern=_ID)
    department: str
    region: str
    account_type: Literal["standard", "privileged", "service", "break_glass"] = "standard"


class IndicatorIn(Strict):
    value: str = Field(max_length=256)
    ioc_type: Literal["ip", "domain", "sha256", "url"]
    confidence: float = Field(ge=0, le=1)
    expires_at: datetime | None = None
    source_feed: str = "internal"


class FeedbackIn(Strict):
    alert_id: str = Field(pattern=_ID)
    verdict: Literal["true_positive", "false_positive", "benign_true_positive"]
    analyst_id: str = Field(pattern=_ID)
    timestamp: datetime
    note: str = Field(default="", max_length=1000)


class ToolObservationIn(Strict):
    tool: str
    target: str
    timestamp: datetime
    output: dict[str, Any]


class Observation(Strict):
    """One raw observation as submitted to the agent (API / scenario files)."""
    kind: Literal["alert", "auth_event", "asset", "user", "indicator", "feedback", "tool_result"]
    source: Literal["siem", "edr", "ids", "auth_log", "asset_inventory", "directory",
                    "threat_intel", "analyst", "tool"]
    data: dict[str, Any]


# ---------------------------------------------------------------- stage 1: PERCEPT
class ValidationStatus(str, Enum):
    VALID = "valid"
    DEGRADED = "degraded"      # usable, but quality issues were found
    REJECTED = "rejected"      # failed schema / quality gate; never reaches the KB


class Percept(Strict):
    percept_id: str
    kind: str
    source: str
    timestamp: datetime | None
    received_at: datetime
    quality: float = Field(ge=0, le=1)
    validation: ValidationStatus
    issues: list[str] = Field(default_factory=list)
    content_flags: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- stage 2: PROPOSITION
class Atom(Strict):
    predicate: str = Field(pattern=r"^[A-Z][A-Za-z]*$")
    args: tuple[str, ...]
    positive: bool = True

    def __str__(self) -> str:
        s = f"{self.predicate}({', '.join(self.args)})"
        return s if self.positive else f"¬{s}"

    @property
    def key(self) -> tuple[str, tuple[str, ...], bool]:
        return (self.predicate, self.args, self.positive)

    def negate(self) -> "Atom":
        return Atom(predicate=self.predicate, args=self.args, positive=not self.positive)


class Proposition(Strict):
    prop_id: str
    atom: Atom
    derived_from: list[str]
    confidence: float = Field(ge=0, le=1)
    testable_by: str


class Contradiction(Strict):
    statement: str
    supporting: list[str]
    opposing: list[str]
    description: str


class PropositionSet(Strict):
    propositions: list[Proposition]
    features: dict[str, dict[str, float]]   # alert_id -> model-ready feature vector
    contradictions: list[Contradiction]


# ---------------------------------------------------------------- stage 3: KNOWLEDGE
class KnowledgeFact(Strict):
    fact_id: str
    atom: Atom
    kind: Literal["observed", "derived", "background", "ontology"]
    confidence: float = Field(ge=0, le=1)
    provenance: list[str]
    asserted_at: datetime


class KnowledgeRule(Strict):
    rule_id: str
    body: list[Atom]
    head: Atom
    description: str
    provenance: str

    @field_validator("body")
    @classmethod
    def _non_empty(cls, v: list[Atom]) -> list[Atom]:
        if not v:
            raise ValueError("rule body must be non-empty")
        return v


class ModelMetadata(Strict):
    name: str
    version: str
    kind: str
    trained_at: str
    metrics: dict[str, float]
    features: list[str]
    artifact_sha256: str | None = None


# ---------------------------------------------------------------- stage 4: INFERENCE
InferenceMethod = Literal[
    "forward_chaining", "bayesian_network", "hmm_filter", "kalman_filter", "ml_model",
    "astar_search", "alpha_beta", "rag_retrieval", "incident_clustering", "risk_fusion",
    "rl_advisor", "fallback",
]


class InferenceStep(Strict):
    step_id: str
    method: InferenceMethod
    subject: str
    inputs: list[str]
    output: str
    value: float | None = None
    rationale: str


class RiskAssessment(Strict):
    asset_id: str
    p_compromise_bayes: float
    p_true_positive_ml: float | None
    hmm_stage: str
    hmm_stage_prob: float
    kalman_z: float
    blast_radius: float
    criticality: int
    risk_score: float
    rank: int = 0
    alerts: list[str]


class Incident(Strict):
    incident_id: str
    incident_type: str
    assets: list[str]
    users: list[str]
    alerts: list[str]
    confidence: float
    stage: str


class InferenceTrace(Strict):
    trace_id: str
    steps: list[InferenceStep]
    derived_facts: list[str]
    risks: list[RiskAssessment]
    incidents: list[Incident]
    retrieved_playbooks: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- stage 5: ENTAILMENT
class EntailmentStatus(str, Enum):
    ENTAILED = "entailed"            # logically provable from the KB, no conflict on the proof
    CONTRADICTED = "contradicted"    # negation provable, or proof rests on conflicting facts
    UNCERTAIN = "uncertain"          # partial / probabilistic support only - NOT proof
    UNKNOWN = "unknown"              # no relevant evidence


class EntailmentResult(Strict):
    query: str
    status: EntailmentStatus
    proof: list[str] = Field(default_factory=list)
    counter_proof: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    probabilistic_support: float | None = None
    sufficiently_supported: bool = False
    rationale: str


# ---------------------------------------------------------------- stage 6: ACTION
class ActionType(str, Enum):
    MONITOR = "monitor"
    ENRICH = "enrich"
    OPEN_TICKET = "open_ticket"
    ADD_WATCHLIST = "add_watchlist"
    NOTIFY_ANALYST = "notify_analyst"
    ESCALATE = "escalate"
    ISOLATE_HOST = "isolate_host"
    DISABLE_ACCOUNT = "disable_account"
    BLOCK_IP = "block_ip"
    ROTATE_CREDENTIALS = "rotate_credentials"
    ABSTAIN = "abstain"


class ProposalStatus(str, Enum):
    AUTO_APPROVED = "auto_approved"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    EXECUTED = "executed"
    FAILED = "failed"
    RECOMMENDED = "recommended"     # advisory only, no tool call


class ActionProposal(Strict):
    proposal_id: str
    case_id: str
    action: ActionType
    tool: str | None
    target: str
    params: dict[str, Any] = Field(default_factory=dict)
    justification: list[str]
    entailment_status: EntailmentStatus | None = None
    utility: float
    confidence: float = Field(ge=0, le=1)
    disruptive: bool
    requires_approval: bool
    status: ProposalStatus
    blocked_reasons: list[str] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)
    schedule: dict[str, Any] | None = None
    idempotency_key: str


class ToolResult(Strict):
    call_id: str
    tool: str
    proposal_id: str
    ok: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float
    idempotent_replay: bool = False


class AuditEvent(Strict):
    seq: int
    event_id: str
    ts: datetime
    actor: str
    event_type: str
    case_id: str | None
    details: dict[str, Any]
    prev_hash: str
    hash: str


class CaseResult(Strict):
    case_id: str
    scenario: str | None
    iterations: int
    halted: bool
    halt_reason: str | None = None
    percepts: list[Percept]
    propositions: list[Proposition]
    contradictions: list[Contradiction]
    knowledge_used: list[str]
    inference: InferenceTrace
    entailments: list[EntailmentResult]
    proposals: list[ActionProposal]
    tool_results: list[ToolResult]
    rl_advice: dict[str, str] = Field(default_factory=dict)
    summary: str
    run_metadata: dict[str, Any]
