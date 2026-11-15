# Ardook Data: Organization Profile (Source of Truth)

Facts about **Ardook Data**, used to find grant leads for it. Ardook Data is separate from Ardook, the laundry social enterprise in `ardook-profile.md`. Select this profile with `python src/run_pipeline.py --profile ardook-data`.

- **Why this file exists:** the grant spreadsheet the sponsor gave us ("Ardook Data Grant Recommendations", rows 1–37) and the TA's Week 8 hidden test both judge grants for Ardook Data, not for the laundry business.
- **Last verified:** 2026-11-12.
- **Sources:** every fact cites a row of the sponsor's spreadsheet ([S] row N, in the "Why It Fits Ardook Data", "Key Concern / Disqualifier" and "Core Requirements" columns), or the Ardook profile. No public source on Ardook Data was found (web search, 2026-10-29), so everything here is the sponsor's own description. Facts not stated there are in [Open items](#open-items).
- **Not used:** the TA's hidden-test answer file was not used to write this profile.

## At a glance

| | |
| --- | --- |
| **Name** | Ardook Data |
| **What it is** | A woman-led AI / physical-AI / data venture with a social-impact dimension [S] row 1 |
| **Stage** | Early-stage: a new entity working toward a defined MVP [S] rows 2, 7, 10, 26 |
| **Legal form** | For-profit company: nonprofit-only programs are flagged as disqualifying [S] rows 9, 13, 24 |
| **Where** | United States: an "early-stage woman-led U.S. venture" [S] row 4, "Bay Area" [S] row 14, "California-based" [S] row 15 |
| **Founder** | Woman-led [S] rows 1, 14, 26; a "woman/mother founder" [S] row 37 |
| **Revenue** | Little or none yet: operating-history and revenue minimums "may disqualify a new Ardook Data entity" [S] row 7; pre-revenue programs fit [S] row 4 |

## What it builds

An AI/data product (described as agentic AI, AI agents, AI/SaaS/data) [S] rows 1, 26, 27, 28. Its technical novelty is not yet proven: deep-tech programs need it substantiated, and "ordinary SaaS / use of existing LLM APIs" is not enough for R&D grants [S] rows 1, 12.

## Mission and the problem it solves

Ardook Data connects its AI/data model to the proven impact of Ardook, the laundry social enterprise employing mothers of children with disabilities in Armenia (see `ardook-profile.md`). Its impact case is the economic empowerment of marginalised women [S] row 11. The sponsor rates women-founder, early-stage, impact and AI programs as its best fits [S] rows 1, 2, 10, 14, 20.

## Theme ratings (used for Fit)

The same grant-focus categories as Ardook's profile, rated for Ardook Data. Ratings follow the sponsor's Fit ratings in rows 1–37, which makes them optimistic when scored on those same rows.

| Rating | Categories |
| --- | --- |
| Good | AI and technology products ([S] rows 1, 14, 16, 19, 26) · women entrepreneurs in any industry (rows 3, 5, 6, 20) · social innovation and social ventures (rows 2, 10) · women's economic empowerment and decent work (row 11) |
| Partial | small business growth (row 7) · family well-being (row 8) · nonprofit capacity building (row 9) · politics and policy (rows 13, 30) · science and research (row 23) · jobs or income for mothers and caregivers (row 37) · and the rest of Ardook's mission themes, linked through Ardook's impact (row 11): disability inclusion, home-based or flexible work, social enterprises that employ women, reducing inequality, gender equality and women's rights, poverty reduction and community development, workforce development and job training, eco-friendly business, social justice and equity, women's leadership development |
| Bad | education and scholarships · health and medicine · violence prevention · arts and culture · environment and conservation · LGBTQ+ programs · disaster relief and emergency response · legal advocacy · sports, religion or infrastructure (no row suggests these fit an AI/data venture) |

## Lead rules

1. **Open to US companies,** plus global calls. The sponsor also rated an EU Horizon Europe program Excellent (Women TechEU, [S] row 1), so Horizon Europe calls count too.
2. **AI and tech programs are leads.** Deep-tech and R&D programs depend on proving technical novelty (Open item 2).
3. **Nonprofit-only programs are not leads** (for-profit).
4. **Exclude only on confirmed facts;** unknown ones keep a grant as a conditional lead.

### Machine-readable facts

```yaml
name: Ardook Data
labeled_sheet_rows: [1, 37]           # the spreadsheet rows the sponsor rated for Ardook Data
operating_countries: [United States]  # [S] rows 4, 14, 15
planned_countries: []
operating_places:                     # for grants limited to part of a country
  - California                        # [S] row 15
  - Bay Area                          # [S] row 14
  - San Francisco
planned_places: []
eligible_regions:
  - Horizon Europe                    # [S] row 1 (Women TechEU rated Excellent)
woman_led: true                       # [S] rows 1, 14, 26
women_ownership_percent: null         # Open item 3
legal_form: for-profit company        # [S] rows 9, 13, 24
established_year: null                # a new entity [S] row 7; exact date unknown (Open item 1)
launched_year: null
stage: new                            # early-stage, pre-MVP/MVP [S] rows 2, 7, 10: meets any "under N years" limit
builds_ai: true                       # [S] rows 1, 12, 14, 26
annual_revenue_usd: null              # little or none yet [S] rows 4, 7
not_eligible_as:                      # a company, none of these
  - student
  - university or school
  - government agency
  - hospital or clinic
accepted_funding_types: null          # credits/ecosystem programs are worth it [S] row 16; loans/equity unknown
founder_identity: null                # only "woman / mother founder" is known [S] rows 1, 37
already_applied:                      # "Applied? = Yes" in the sponsor's sheet
  - Women TechEU
  - Tara's Circle
  - Her Agenda Breakthrough Grant
  - EchoHer Founder Fellowship
theme_categories:
  good:
    - AI and technology products
    - women entrepreneurs in any industry
    - social innovation and social ventures
    - women's economic empowerment and decent work
  partial:
    - small business growth
    - family well-being
    - nonprofit capacity building
    - politics and policy
    - science and research
    - jobs or income for mothers and caregivers
    - disability inclusion
    - home-based or flexible work
    - social enterprises that employ women
    - reducing inequality
    - gender equality and women's rights
    - poverty reduction and community development
    - workforce development and job training
    - eco-friendly business
    - social justice and equity
    - women's leadership development
  bad:
    - education and scholarships
    - health and medicine
    - violence prevention
    - arts and culture
    - environment and conservation
    - LGBTQ+ programs
    - disaster relief and emergency response
    - legal advocacy
    - sports, religion or infrastructure
```

## Open items

1. **Registration:** where and when is Ardook Data registered (US state? also in Armenia?), and how does it relate legally to Ardook?
2. **Technical novelty:** is the AI/data technology novel enough for deep-tech and R&D programs (Women TechEU, NSF SBIR, Cartier)? [S] rows 1, 12, 31
3. **Ownership:** the women ownership share (programs ask for ≥25%, ≥33%, ≥50%).
4. **Revenue:** current revenue (several programs set minimums or caps).
5. **Funding types:** whether loans and equity count as leads.

## Sources

- [S] "Ardook Data Grant Recommendations" spreadsheet provided by the sponsor (`Copy of Ardook_Data_All_Grant_Recommendations_UPDATED_v3.xlsx.xlsm`, Grant Tracker rows 1–37).
- `docs/ardook-profile.md` for Ardook, the laundry social enterprise.
