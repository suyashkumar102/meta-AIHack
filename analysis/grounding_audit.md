# Grounding Audit For Taxonomy And Similarity Decisions

> Internal note for the roadmap work originally planned for April 5, 2026.
> Reviewed on April 3, 2026 and pulled forward ahead of schedule.

## Goal

Ground the current ticket taxonomy and the limited partial-credit policy against real public IT-support data without turning external datasets into a runtime dependency.

## Sources Reviewed

1. [Classification of IT Support Tickets](https://zenodo.org/records/7648117)
   - Zenodo dataset with 2,229 manually classified support tickets.
   - Dataset description says the tickets were classified by three IT support professionals.
   - The public preview exposes seven coarse categories: `Fileservice`, `Support general`, `Software`, `O365`, `Active Directory`, `Computer-Services`, and `EOL`.

2. [Semantic Similarity of IT Support Tickets](https://zenodo.org/records/7426225)
   - Zenodo dataset with 300 ticket pairs manually labeled for semantic similarity.
   - The description says three IT support professionals performed the labeling.
   - This is the best direct grounding for keeping similarity explicit and limited instead of treating the whole label space as fuzzy.

3. [MSDialog dataset page](https://ciir.cs.umass.edu/downloads/msdialog/)
   - Technical-support dialog corpus drawn from Microsoft Community.
   - The site reports 35,000 dialogs in `MSDialog-Complete` and 2,199 labeled dialogs with 10,020 utterances in `MSDialog-Intent`.
   - This grounds our use of follow-up cases, clarification-heavy threads, and helpdesk-style conversational language.

## Mapping Principle

The external datasets validate that real IT support traffic mixes access problems, software incidents, generic support questions, procurement-like requests, and multi-turn follow-ups. Our label set is more operational than the public category sets, so the mappings below are judgment calls based on source descriptions and public previews rather than exact label equivalence.

## Grounding Examples

1. Active Directory lockout, MFA trouble, or password reset -> `identity_access` -> exact-match dominant, with `onboarding` as the only defensible adjacent label when the request is really about new-user provisioning.
2. New hire account setup or contractor access provisioning -> `onboarding` -> partial-credit adjacent to `identity_access`, because both can surface as account enablement work before ownership is fully resolved.
3. Office or application crash, timeout, webhook failure, or migration-script breakage -> `application_support` -> partial-credit adjacent to `feature_request` only when the report reads like a capability gap rather than a break/fix issue.
4. Feature wishlist or export-format enhancement request -> `feature_request` -> partial-credit adjacent to `application_support` only when the user reports the missing capability as if it were a defect.
5. Vendor-evaluation question, demo request, or quote request -> `service_request` -> partial-credit adjacent to `general_inquiry` when the request is still exploratory rather than a committed operational action.
6. Seat expansion or provisioning-style commercial request -> `service_request` -> partial-credit adjacent to `billing_license` when procurement and account-admin signals are mixed in the same ticket.
7. Refund, invoice discrepancy, subscription cancellation, or payment-admin issue -> `billing_license` -> partial-credit adjacent to `service_request` only in commercial admin cases that overlap with a procurement or seat-change request.
8. Broad capability question or lightweight product clarification -> `general_inquiry` -> partial-credit adjacent to `service_request` or `feature_request` when the request is vague enough to look like either evaluation or roadmap feedback.
9. Spam lure or credential-phishing message sent to the inbox -> `spam_phishing` -> partial-credit adjacent to `security_compliance` only for security-themed inbound items, not for normal access or software tickets.
10. GDPR deletion request, DPA request, audit finding, or mandatory MFA policy notice -> `security_compliance` -> exact-match dominant, with very limited adjacency to `spam_phishing` for suspicious security reports and a low-confidence edge to `billing_license` only in contractual paperwork contexts.
11. Reopened outage thread or repeated bug report escalation -> `application_support` -> exact-match dominant; the main change across turns is usually `priority`, not `issue_type`.
12. Repeated lockout complaint or suspension follow-up -> `identity_access` -> exact-match dominant; follow-up behavior is grounded by MSDialog-style multi-turn support flow rather than by adding new label fuzziness.

## Review Of Current Similarity Pairs

The current `ISSUE_TYPE_SIMILARITY` map stays intentionally small. The defensible themes are:

- `billing_license` <-> `service_request`: commercial admin and procurement requests can overlap before the owning team is clear.
- `application_support` <-> `identity_access`: SSO and login failures can initially look like either app failure or access failure.
- `application_support` <-> `feature_request`: some users describe missing functionality in bug-report language.
- `onboarding` <-> `identity_access`: provisioning and account enablement are adjacent in real helpdesk traffic.
- `general_inquiry` <-> `feature_request`: vague product questions can blur into roadmap requests.
- `general_inquiry` <-> `service_request`: vendor-evaluation and exploratory capability questions often overlap.
- `spam_phishing` <-> `security_compliance`: both are security-facing, but they should stay separate from normal access or app-routing labels.
- `security_compliance` <-> `billing_license`: kept only as a very low-score edge for contract and paperwork overlap; this is the weakest current pair and should not be expanded further without ticket-level evidence.

## Candidate Expansions Reviewed And Rejected

These pairs were reviewed during the April 5 roadmap pass and are intentionally not being added:

- `onboarding` <-> `service_request`: both can involve setup, but the owning teams and next actions diverge too quickly.
- `feature_request` <-> `service_request`: roadmap asks and procurement actions are operationally different.
- `security_compliance` <-> `identity_access`: policy obligations may mention accounts, but the compliance workflow is distinct from user access support.
- `billing_license` <-> `identity_access`: nonpayment or suspension can mention lockout symptoms, but the root-cause owner is different.
- `application_support` <-> `billing_license`: mixed commercial and outage narratives exist, but broad partial credit here would blur incident handling too much.

## Decision

No new issue-type similarity pairs should be added from this review.

The safest grounded position is:

- keep the current limited similarity map,
- rely on exact-match scoring for most wrong labels,
- let `priority`, `assignment_group`, and `resolution_action` keep the hard-task routing signal crisp.
