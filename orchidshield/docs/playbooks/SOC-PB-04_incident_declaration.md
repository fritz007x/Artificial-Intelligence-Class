# SOC-PB-04 Incident declaration and escalation
Declare an incident when a true positive alert implies a compromised host or account. Group alerts that
share a host or user within the same time window into one incident. Contradictory evidence such as an
analyst false positive verdict against an indicator match must be escalated to a human analyst; do not
contain on contradicted conclusions. Uncertain cases are enriched and monitored, not contained.
