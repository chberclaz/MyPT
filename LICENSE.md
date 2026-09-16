# Licensing

MyPT is dual-licensed.

- **Open source:** GNU Affero General Public License, version 3 (AGPL-3.0-only). The full text is in
  [`LICENSE`](LICENSE). This applies to everything in this repository unless a file states otherwise.
- **Commercial:** a separate, negotiated licence that removes the AGPL's copyleft obligations.
  See [Commercial licensing](#3-commercial-licensing) below.

You may use MyPT under either licence. You choose; you do not need permission to choose AGPL-3.0.

**In short:** use it, study it, modify it, deploy it — free, forever, under AGPL-3.0. If you need to
keep your modifications closed, or you want trained checkpoints and support, talk to me about a
commercial licence.

**Copyright © 2026 Christian Berclaz, Switzerland.**

---

## 1. What is covered

This repository contains the MyPT pipeline: training and data-preparation scripts, the evaluation
harness, the lineage tooling, configuration files, and documentation. All of it is AGPL-3.0 unless a
file header says otherwise.

**Not in this repository and not covered by this licence:**

| Artifact | Status |
|---|---|
| Trained model weights and checkpoints | Distributed separately under commercial terms. |
| Generated SFT corpora and composition manifests | Not distributed. |
| Gold evaluation sets | Not distributed. |
| Tuned curriculum mixes and phase weights | Not published. |
| Support, onboarding, warranties | Not included at the open-source tier. |

Third-party datasets referenced by the pipeline are **not** redistributed here and are **not**
covered by this licence. Each carries its own terms — see the data provenance section of the
README. Verify the licence of any dataset before using it, especially commercially.

Third-party dependencies retain their own licences. This licence does not extend to them.

---

## 2. Using MyPT under AGPL-3.0

In plain language, and without replacing the licence text itself:

- You may run, study, modify, and distribute MyPT freely, including commercially.
- If you distribute MyPT or a modified version, you must provide the complete corresponding source
  under AGPL-3.0.
- **Section 13 is what distinguishes AGPL from GPL:** if you modify MyPT and let users interact with
  it over a network, those users must be offered the corresponding source of your modified version —
  even though you never distributed a copy to them.
- The licence carries no warranty and no liability. See sections 15–17 of the licence text.

Purely internal use, with no modified version exposed to outside users over a network, does not
trigger the source-offer obligation. Whether a specific deployment does is a question for your own
counsel, not for this document.

### Applying the notice

New source files should carry the standard header:

```
Copyright (C) 2026 Christian Berclaz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```

`SPDX-License-Identifier: AGPL-3.0-only` is an acceptable short form.

---

## 3. Commercial licensing

The AGPL is deliberate. It keeps MyPT open for people who will keep it open, and it makes the
commercial licence meaningful for organisations that cannot accept copyleft — which, in practice,
is most regulated enterprises.

A commercial licence is appropriate if you want to:

- embed MyPT in a proprietary product or internal system without releasing your modifications,
- offer a network service built on a modified MyPT without the section 13 source offer,
- receive trained checkpoints, tuned curriculum configurations, or evaluation sets,
- receive support, onboarding, or contractual warranties.

### How the term works

Commercial licences are annual. What that means concretely, so there is no ambiguity at renewal:

- The licence grant for **each version delivered during the term is perpetual for that version**.
  If the term lapses, deployments you already run do not become unlicensed and do not have to be
  shut down.
- What lapses is the right to **new** versions, new checkpoints, support, and any attestation or
  documentation service in the agreement.
- AGPL rights are never revoked and never depend on a commercial term. Anyone may fall back to
  AGPL-3.0 at any time.

This structure exists so that a customer's production system is never hostage to a renewal
negotiation. It is a deliberate consequence of the sovereignty principle, not an oversight.

Commercial licences are negotiated individually. Contact: **ch.berclaz@gmail.com**

---

## 4. Contributions

**Inbound licensing terms.** By submitting a contribution — a pull request, patch, or any other
material — you represent and agree that:

1. You are the author of the contribution, or are otherwise entitled to submit it, and it does not
   knowingly infringe any third party's rights.
2. Where your employer holds rights to work you create — which varies by jurisdiction and contract —
   you have permission to submit the contribution, or those rights do not apply to it.
3. You license the contribution under **AGPL-3.0-only**, and
4. You **additionally grant the copyright holder a perpetual, worldwide, irrevocable, royalty-free
   right to license your contribution under other terms, including proprietary commercial terms**,
   without further notice or compensation.

Point 4 is what makes dual licensing possible. Without it, a contribution can only ever be AGPL, and
the commercial licence could not cover the resulting code.

You retain copyright in your contribution. You are granting a licence, not assigning ownership.

Submitting a pull request constitutes agreement to the above. Where a `Signed-off-by:` line is
present, it additionally certifies the Developer Certificate of Origin 1.1
(<https://developercertificate.org/>) — note that the DCO alone does **not** cover point 4, which is
why these terms are stated separately.

Contributions of any size are welcome under these terms. If you cannot agree to point 4 — for
instance because your employer forbids it — open an issue describing the change instead; the
suggestion is still useful.

---

## 5. Trademark

The name "MyPT" and any associated logo are not licensed under AGPL-3.0. The AGPL grants copyright
permissions, not trademark permissions.

A fork must be distinguishable by name. You may state accurately that your work is derived from or
based on MyPT; you may not use the name in a way that suggests the fork is the official project or
is endorsed by it.

The name is currently **unregistered**. This section is therefore a statement of intent and a request
for good practice, not an assertion of registered rights.

---

## 6. No warranty and liability

This document explains the licensing arrangement. It is not the licence — the AGPL-3.0 text in
[`LICENSE`](LICENSE) governs, and where the two differ, that text prevails.

MyPT is provided as is, without warranty of any kind, to the extent permitted by applicable law.

Under Swiss law, liability for gross negligence and intentional harm cannot be excluded by contract
(Art. 100 CO). Nothing in this document attempts to exclude it. Commercial agreements may define
warranties and liability separately; where they do, the commercial agreement governs for that
customer.

---

## 7. Governing law

This document and the AGPL grant described here are governed by Swiss law, to the extent that a
governing-law clause applies to a unilateral licence grant. Commercial agreements state their own
governing law and venue.
