# SOC-PB-05 Containment and segmentation
Containment actions (isolate host, disable account, block ip, rotate credentials) are disruptive and
require human approval. Never isolate a domain controller: instead block the external source ip,
disable the compromised account and isolate the adjacent workstation or server that provides the
attack path (segmentation). Never touch the SIEM. At most two isolations per incident without
incident commander review. Block only external ip addresses at the perimeter.
