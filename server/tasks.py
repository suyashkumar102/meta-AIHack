from __future__ import annotations

import json
from pathlib import Path

from models import HelpdeskTicketRecord
from vocabulary import TASK_IDS


TASKS = {
    1: {
        "id": 1,
        "name": "Guided Full Routing",
        "difficulty": "easy",
        "instructions": (
            "Perform full helpdesk routing by selecting issue type, priority, "
            "assignment group, and resolution action. Easy-task episodes keep the "
            "ticket text mostly visible and focus on grounded single-ticket routing."
        ),
        "allowed_fields": [
            "issue_type",
            "priority",
            "assignment_group",
            "resolution_action",
        ],
    },
    2: {
        "id": 2,
        "name": "Contextual Full Routing",
        "difficulty": "medium",
        "instructions": (
            "Perform full helpdesk routing with partial observability and moderate "
            "queue carry-over. Some tickets hide related-case, requester-history, "
            "or cluster-coordination details until you investigate or request more "
            "information, and medium episodes can also require deferral or coherent "
            "handling across linked tickets in the same queue."
        ),
        "allowed_fields": [
            "issue_type",
            "priority",
            "assignment_group",
            "resolution_action",
        ],
    },
    3: {
        "id": 3,
        "name": "Adaptive Queue Routing",
        "difficulty": "hard",
        "instructions": (
            "Perform full helpdesk routing by selecting the best issue type, "
            "priority, assignment group, and resolution action for the ticket. "
            "Use any ambiguity notes, related-ticket previews, queue-capacity "
            "forecasts, and planning state when present. "
            "Some hard tickets intentionally hide decisive routing context until "
            "you investigate with the available tools, and some hard episodes also "
            "require queue-level capacity planning, deferrals, incident management, "
            "and recovery from downstream follow-up tickets."
        ),
        "allowed_fields": [
            "issue_type",
            "priority",
            "assignment_group",
            "resolution_action",
        ],
    },
}


PLANNING_ROUTE_UPDATES: dict[str, dict] = {
    "ticket-022": {
        "service_cluster_id": "commerce_outage_recovery",
        "planning_note": (
            "If the application queue is saturated, billing operations can own the "
            "customer-facing charge review as a lower-fidelity fallback while the bug "
            "investigation continues separately."
        ),
        "customer_update_note": (
            "Finance confirmed the unexpected charge landed immediately after the "
            "integration outage and wants one accountable owner today."
        ),
        "alternate_issue_type": "billing_license",
        "alternate_assignment_group": "license_ops",
        "alternate_resolution_action": "assign",
        "alternate_route_score_multiplier": 0.74,
    },
    "ticket-027": {
        "planning_note": (
            "If procurement capacity is available, treat this like a commercial review. "
            "If not, a lightweight service-desk acknowledgement is still acceptable."
        ),
        "alternate_issue_type": "service_request",
        "alternate_priority": "medium",
        "alternate_assignment_group": "procurement",
        "alternate_resolution_action": "assign",
        "alternate_route_score_multiplier": 0.92,
    },
    "ticket-029": {
        "planning_note": (
            "Seat expansion is the preferred route, but license operations can still "
            "handle the prorating clarification when procurement is the bottleneck."
        ),
        "customer_update_note": (
            "The requester clarified that the blocker is both the seat increase and "
            "the unexpected prorating language on the quote."
        ),
        "alternate_issue_type": "billing_license",
        "alternate_assignment_group": "license_ops",
        "alternate_resolution_action": "fulfill",
        "alternate_route_score_multiplier": 0.82,
    },
    "ticket-040": {
        "planning_note": (
            "The request can be treated either as roadmap feedback or as a support "
            "escalation if the operational impact is emphasized."
        ),
        "customer_update_note": (
            "The requester says the missing behavior is now blocking a customer "
            "rollout, so this may need operational ownership rather than product triage."
        ),
        "alternate_issue_type": "application_support",
        "alternate_priority": "high",
        "alternate_resolution_action": "escalate",
        "alternate_route_score_multiplier": 0.76,
    },
    "ticket-046": {
        "service_cluster_id": "atlasbank_lockout_bridge",
    },
    "ticket-047": {
        "service_cluster_id": "bluequarry_launch_readiness",
        "planning_note": (
            "The preferred route is an immediate service-desk extension, but the "
            "commercial owner can take it if operational fulfillment capacity is exhausted."
        ),
        "alternate_assignment_group": "procurement",
        "alternate_resolution_action": "assign",
        "alternate_route_score_multiplier": 0.78,
    },
    "ticket-048": {
        "planning_note": (
            "This belongs with procurement when commercial reviewers are available, "
            "but a generic service-desk acknowledgement is an acceptable fallback."
        ),
        "alternate_assignment_group": "service_desk",
        "alternate_resolution_action": "acknowledge",
        "alternate_route_score_multiplier": 0.9,
    },
    "ticket-050": {
        "service_cluster_id": "merger_onboarding_wave",
        "planning_note": (
            "Central coordination is preferred. If service-desk capacity is depleted, "
            "onboarding operations can still run a reduced fulfillment path."
        ),
        "alternate_priority": "medium",
        "alternate_assignment_group": "onboarding_ops",
        "alternate_resolution_action": "fulfill",
        "alternate_route_score_multiplier": 0.84,
    },
    "ticket-051": {
        "planning_note": (
            "Commercial procurement owns the contract amendment, but this can also "
            "be treated as a service request when the commercial queue needs triage."
        ),
        "alternate_issue_type": "service_request",
        "alternate_route_score_multiplier": 0.83,
    },
    "ticket-052": {
        "service_cluster_id": "clientgrid_evidence_renewal",
    },
    "ticket-053": {
        "planning_note": (
            "Security scheduling is ideal, but a compliance acknowledgement is still "
            "acceptable when the security team only needs to confirm the process."
        ),
        "customer_update_note": (
            "The requester clarified they mainly need confirmed ownership and a date "
            "for the review, not the review itself right now."
        ),
        "alternate_issue_type": "security_compliance",
        "alternate_resolution_action": "acknowledge",
        "alternate_route_score_multiplier": 0.8,
    },
    "ticket-054": {
        "planning_note": (
            "License operations can fulfill the archive request directly. If that queue "
            "is saturated, service desk can acknowledge and queue the retrieval."
        ),
        "alternate_assignment_group": "service_desk",
        "alternate_resolution_action": "acknowledge",
        "alternate_route_score_multiplier": 0.9,
    },
    "ticket-038": {
        "service_cluster_id": "commerce_outage_recovery",
    },
}


CURATED_EXPANSION_RECORDS: list[dict] = [
    {
        "ticket_id": "ticket-056",
        "title": "Vendor DPA redlines need an owner before pricing sign-off",
        "requester": "procurement@harborcompliance.io",
        "description": (
            "Commercial review is already moving, but the team needs to know who owns "
            "the vendor DPA redlines before pricing can be approved."
        ),
        "issue_type": "general_inquiry",
        "priority": "medium",
        "assignment_group": "procurement",
        "resolution_action": "assign",
        "planning_note": (
            "Procurement is preferred, but service desk can acknowledge and route the "
            "questionnaire logistics if the commercial queue is saturated."
        ),
        "alternate_assignment_group": "service_desk",
        "alternate_resolution_action": "acknowledge",
        "alternate_route_score_multiplier": 0.9,
    },
    {
        "ticket_id": "ticket-057",
        "title": "Board audit packet needs a timeline for the privileged-account lockout",
        "requester": "security-ops@atlasbank.io",
        "description": (
            "Following up on ticket-046. The board pack needs a timeline and ownership "
            "summary for the privileged admin lockout before tomorrow morning."
        ),
        "issue_type": "identity_access",
        "priority": "high",
        "assignment_group": "security_team",
        "resolution_action": "escalate",
        "related_ticket_id": "ticket-046",
        "planning_note": (
            "Security still owns the privileged-access review, but service desk can "
            "collect chronology and prepare the packet if the security queue is jammed."
        ),
        "customer_update_note": (
            "Executives want a single incident bridge owner before the board packet is sent."
        ),
        "incident_recommended": True,
        "service_cluster_id": "atlasbank_lockout_bridge",
        "alternate_assignment_group": "service_desk",
        "alternate_resolution_action": "assign",
        "alternate_route_score_multiplier": 0.72,
    },
    {
        "ticket_id": "ticket-058",
        "title": "Temporary contractor extension during an onboarding surge",
        "requester": "hr@talentbridge.co",
        "description": (
            "A contractor start date slipped by two weeks and the account needs to stay "
            "active while the onboarding backlog is already full."
        ),
        "issue_type": "onboarding",
        "priority": "medium",
        "assignment_group": "service_desk",
        "resolution_action": "assign",
        "planning_note": (
            "Service desk is preferred for cross-team coordination. If coordination "
            "capacity is exhausted, onboarding operations can fulfill the extension directly."
        ),
        "service_cluster_id": "merger_onboarding_wave",
        "alternate_assignment_group": "onboarding_ops",
        "alternate_resolution_action": "fulfill",
        "alternate_route_score_multiplier": 0.85,
    },
    {
        "ticket_id": "ticket-059",
        "title": "Archived invoice packet plus quarter-close clarification",
        "requester": "boardops@silverpine.com",
        "description": (
            "Finance needs archived invoice PDFs plus a quick note explaining whether any "
            "quarter-close adjustments are still pending."
        ),
        "issue_type": "general_inquiry",
        "priority": "medium",
        "assignment_group": "license_ops",
        "resolution_action": "fulfill",
        "planning_note": (
            "Invoice operations can fulfill directly. If that queue is constrained, "
            "service desk can acknowledge and schedule the retrieval."
        ),
        "service_cluster_id": "commerce_outage_recovery",
        "alternate_assignment_group": "service_desk",
        "alternate_resolution_action": "acknowledge",
        "alternate_route_score_multiplier": 0.88,
    },
    {
        "ticket_id": "ticket-060",
        "title": "Re: Temporary sandbox extension for the signed pilot",
        "requester": "solutions@bluequarry.io",
        "description": (
            "Following up on ticket-047. The customer launch rehearsal is tomorrow, so the "
            "sandbox extension needs either immediate execution or a commercial owner to unblock it."
        ),
        "issue_type": "service_request",
        "priority": "high",
        "assignment_group": "service_desk",
        "resolution_action": "escalate",
        "related_ticket_id": "ticket-047",
        "planning_note": (
            "Immediate operational execution is preferred. Procurement can still own the "
            "approval path if service-desk capacity is already depleted."
        ),
        "customer_update_note": (
            "The customer says the launch rehearsal will fail without a same-day answer."
        ),
        "incident_recommended": True,
        "service_cluster_id": "bluequarry_launch_readiness",
        "alternate_assignment_group": "procurement",
        "alternate_resolution_action": "assign",
        "alternate_route_score_multiplier": 0.8,
    },
    {
        "ticket_id": "ticket-061",
        "title": "Risk-exception review is blocking an SSO restore",
        "requester": "identity-risk@sterlingmed.io",
        "description": (
            "Users cannot log in through SSO until a temporary risk exception is approved. "
            "The product team may need logs, but the unblock decision is tied to the review."
        ),
        "issue_type": "identity_access",
        "priority": "critical",
        "assignment_group": "security_team",
        "resolution_action": "escalate",
        "planning_note": (
            "Security owns the final unblock decision. If security is saturated, the "
            "application team can still take the first-response diagnostics path."
        ),
        "customer_update_note": (
            "The identity-risk lead confirmed users remain locked out and wants incident "
            "coordination while the exception is reviewed."
        ),
        "incident_recommended": True,
        "alternate_issue_type": "application_support",
        "alternate_priority": "high",
        "alternate_assignment_group": "application_team",
        "alternate_resolution_action": "escalate",
        "alternate_route_score_multiplier": 0.74,
    },
    {
        "ticket_id": "ticket-062",
        "title": "Need product remediation evidence for a customer security questionnaire",
        "requester": "assurance@clientgrid.com",
        "description": (
            "A customer questionnaire asks for evidence that a previously remediated "
            "application vulnerability is fully closed."
        ),
        "issue_type": "security_compliance",
        "priority": "medium",
        "assignment_group": "application_team",
        "resolution_action": "fulfill",
        "planning_note": (
            "Application engineering is preferred because they hold the remediation artifacts. "
            "Security can still acknowledge the questionnaire and buy time when app capacity is tight."
        ),
        "service_cluster_id": "clientgrid_evidence_renewal",
        "alternate_assignment_group": "security_team",
        "alternate_resolution_action": "acknowledge",
        "alternate_route_score_multiplier": 0.82,
    },
    {
        "ticket_id": "ticket-063",
        "title": "Subsidiary admin training with a seat-transfer request",
        "requester": "enablement@globalcorp.com",
        "description": (
            "A newly acquired subsidiary needs admin training next week and also wants "
            "to transfer existing seats into the parent contract."
        ),
        "issue_type": "service_request",
        "priority": "medium",
        "assignment_group": "procurement",
        "resolution_action": "assign",
        "planning_note": (
            "Procurement owns the commercial transfer. If that queue is overloaded, "
            "onboarding operations can still deliver the training portion first."
        ),
        "alternate_issue_type": "onboarding",
        "alternate_assignment_group": "onboarding_ops",
        "alternate_resolution_action": "fulfill",
        "alternate_route_score_multiplier": 0.78,
    },
    {
        "ticket_id": "ticket-064",
        "title": "Legal-hold export of invoice history",
        "requester": "legalops@northshoreenergy.com",
        "description": (
            "Legal needs invoice history exported for a hold notice. No pricing change is "
            "required, but the request must be acknowledged today."
        ),
        "issue_type": "general_inquiry",
        "priority": "high",
        "assignment_group": "license_ops",
        "resolution_action": "fulfill",
        "planning_note": (
            "License operations can deliver the export. If they are capacity-constrained, "
            "service desk can acknowledge the request and queue the retrieval."
        ),
        "alternate_assignment_group": "service_desk",
        "alternate_resolution_action": "acknowledge",
        "alternate_route_score_multiplier": 0.87,
    },
    {
        "ticket_id": "ticket-065",
        "title": "Cross-functional launch checklist for an acquired support team",
        "requester": "integration@mergerco.com",
        "description": (
            "Twelve support agents from an acquired business need onboarding, mailbox "
            "setup, and a security attestation before Monday."
        ),
        "issue_type": "onboarding",
        "priority": "high",
        "assignment_group": "service_desk",
        "resolution_action": "assign",
        "planning_note": (
            "Central coordination is preferred. If service-desk capacity is exhausted, "
            "onboarding operations can still run a reduced fulfillment path."
        ),
        "service_cluster_id": "merger_onboarding_wave",
        "alternate_priority": "medium",
        "alternate_assignment_group": "onboarding_ops",
        "alternate_resolution_action": "fulfill",
        "alternate_route_score_multiplier": 0.81,
    },
    {
        "ticket_id": "ticket-066",
        "title": "Pilot customer asks who approves a credential-defense allowlist",
        "requester": "pilotops@cruxsystems.io",
        "description": (
            "A pilot customer needs to know who approves an IP allowlist for a credential-"
            "defense control before they continue their test."
        ),
        "issue_type": "general_inquiry",
        "priority": "medium",
        "assignment_group": "security_team",
        "resolution_action": "assign",
        "planning_note": (
            "Security should own the answer when available. If that queue is overloaded, "
            "service desk can acknowledge and route the ownership question."
        ),
        "alternate_assignment_group": "service_desk",
        "alternate_resolution_action": "acknowledge",
        "alternate_route_score_multiplier": 0.84,
    },
    {
        "ticket_id": "ticket-067",
        "title": "Re: Remediation evidence package is now blocking a renewal signature",
        "requester": "assurance@clientgrid.com",
        "description": (
            "Following up on ticket-052. Renewal signature is blocked until the remediation "
            "evidence package is delivered or a commercial owner confirms the delay."
        ),
        "issue_type": "security_compliance",
        "priority": "high",
        "assignment_group": "application_team",
        "resolution_action": "escalate",
        "related_ticket_id": "ticket-052",
        "planning_note": (
            "Application engineering is preferred because they own the evidence. Procurement "
            "can still coordinate the renewal communication if the evidence queue is saturated."
        ),
        "customer_update_note": (
            "Commercial leadership needs one named owner for the blocked renewal before end of day."
        ),
        "service_cluster_id": "clientgrid_evidence_renewal",
        "alternate_issue_type": "service_request",
        "alternate_priority": "medium",
        "alternate_assignment_group": "procurement",
        "alternate_resolution_action": "assign",
        "alternate_route_score_multiplier": 0.76,
    },
    {
        "ticket_id": "ticket-068",
        "title": "Re: Second executive follow-up on the privileged-account lockout bridge",
        "requester": "security-ops@atlasbank.io",
        "description": (
            "Another leadership update landed after ticket-057. Executives want to know "
            "whether one incident owner is already coordinating the privileged-account lockout."
        ),
        "issue_type": "identity_access",
        "priority": "high",
        "assignment_group": "security_team",
        "resolution_action": "escalate",
        "related_ticket_id": "ticket-057",
        "planning_note": (
            "Security is still the preferred owner, but if an incident bridge is already "
            "running, service desk can acknowledge and consolidate the status update."
        ),
        "customer_update_note": (
            "Leadership said they do not want a second parallel workstream for the same lockout."
        ),
        "incident_recommended": True,
        "service_cluster_id": "atlasbank_lockout_bridge",
        "alternate_assignment_group": "service_desk",
        "alternate_resolution_action": "acknowledge",
        "alternate_route_score_multiplier": 0.78,
    },
    {
        "ticket_id": "ticket-069",
        "title": "Commercial sign-off still pending for the bluequarry launch rehearsal",
        "requester": "solutions@bluequarry.io",
        "description": (
            "The sandbox extension is still the blocker, but finance now only needs one "
            "owner to confirm whether commercial approval or operational execution comes next."
        ),
        "issue_type": "service_request",
        "priority": "medium",
        "assignment_group": "procurement",
        "resolution_action": "assign",
        "related_ticket_id": "ticket-060",
        "planning_note": (
            "Procurement owns the commercial answer, but if an earlier bridge is already "
            "active the service desk can acknowledge and link this follow-up into that track."
        ),
        "customer_update_note": (
            "The customer only wants a single coordinated answer instead of separate procurement and support replies."
        ),
        "service_cluster_id": "bluequarry_launch_readiness",
        "alternate_assignment_group": "service_desk",
        "alternate_resolution_action": "acknowledge",
        "alternate_route_score_multiplier": 0.82,
    },
    {
        "ticket_id": "ticket-070",
        "title": "Renewal counsel asks for a named owner on the remediation evidence package",
        "requester": "assurance@clientgrid.com",
        "description": (
            "Legal asked whether the remediation evidence request from ticket-067 is now "
            "covered by an existing owner or needs a fresh commercial escalation."
        ),
        "issue_type": "security_compliance",
        "priority": "high",
        "assignment_group": "application_team",
        "resolution_action": "escalate",
        "related_ticket_id": "ticket-067",
        "planning_note": (
            "Application engineering still owns the evidence, but a coordinated service-desk "
            "acknowledgement is acceptable if the earlier remediation thread already has an owner."
        ),
        "customer_update_note": (
            "Counsel said duplicate answers from product and commercial teams would make the renewal risk look worse."
        ),
        "service_cluster_id": "clientgrid_evidence_renewal",
        "alternate_assignment_group": "service_desk",
        "alternate_resolution_action": "acknowledge",
        "alternate_route_score_multiplier": 0.8,
    },
    {
        "ticket_id": "ticket-071",
        "title": "Access matrix still blocking the merger onboarding wave",
        "requester": "integration@mergerco.com",
        "description": (
            "The onboarding launch from ticket-065 is still blocked because the access matrix "
            "for the acquired support team is incomplete and Monday is getting close."
        ),
        "issue_type": "onboarding",
        "priority": "high",
        "assignment_group": "service_desk",
        "resolution_action": "assign",
        "related_ticket_id": "ticket-065",
        "planning_note": (
            "Central coordination remains preferred. If an earlier owner is already driving the "
            "wave, onboarding operations can fulfill the matrix updates while service desk handles the communications."
        ),
        "customer_update_note": (
            "The integration lead wants one owner for the remaining blocker instead of a fresh handoff."
        ),
        "service_cluster_id": "merger_onboarding_wave",
        "alternate_assignment_group": "onboarding_ops",
        "alternate_resolution_action": "fulfill",
        "alternate_route_score_multiplier": 0.83,
    },
]


def _apply_dataset_enhancements(
    dataset: list[HelpdeskTicketRecord],
) -> list[HelpdeskTicketRecord]:
    enhanced_dataset: list[HelpdeskTicketRecord] = []
    for record in dataset:
        update = PLANNING_ROUTE_UPDATES.get(record.ticket_id)
        enhanced_dataset.append(
            record.model_copy(update=update) if update is not None else record
        )

    seen_ids = {record.ticket_id for record in enhanced_dataset}
    for raw_record in CURATED_EXPANSION_RECORDS:
        ticket_id = str(raw_record["ticket_id"])
        if ticket_id in seen_ids:
            raise ValueError(f"Duplicate ticket_id in curated expansion: {ticket_id}")
        enhanced_dataset.append(HelpdeskTicketRecord.model_validate(raw_record))
        seen_ids.add(ticket_id)
    return enhanced_dataset

assert tuple(TASKS.keys()) == TASK_IDS


def load_dataset() -> list[HelpdeskTicketRecord]:
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "dataset.json"
    # Accept UTF-8 files saved with a BOM, which is common on Windows editors.
    with dataset_path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    dataset = [HelpdeskTicketRecord.model_validate(r) for r in raw]
    return _apply_dataset_enhancements(dataset)


def get_task_definition(task_id: int) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Unsupported task_id: {task_id}")
    return TASKS[task_id]
